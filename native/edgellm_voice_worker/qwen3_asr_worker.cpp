/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{
struct Args
{
    std::string engineDir;
    std::string multimodalEngineDir;
    bool debug{false};
};

// ---------------------------------------------------------------------------
// M3 streaming-ASR worker (design doc §15 v5.2).
//
// The worker is now event-driven on stdin. Each line is a JSON object:
//
//   {"event":"begin","id":<sid>,"sample_rate":16000,
//    "chunk_size_sec":0.5,"unfixed_chunk_num":2,"unfixed_token_num":5,
//    "force_language":null,"context":""}
//   {"event":"chunk","id":<sid>,"mel_path":<path to fp16 mel safetensors>,"last":false}
//   {"event":"end","id":<sid>}
//
// Lines with no `event` field hit the legacy one-shot handler — required
// for byte-equivalent backward compatibility with M2 callers.
//
// Single-session worker: a `begin` arriving while another session is active
// is refused with {"event":"error","error":"session_already_active"}.
// ---------------------------------------------------------------------------
struct AsrSessionState
{
    std::string sessionId;                                   //!< Stable ID emitted by the client at begin.
    std::chrono::steady_clock::time_point lastActivity{};    //!< Updated on every chunk/end touching the slot.
    bool active{false};                                      //!< True between begin and end.

    // §15.1 streaming state — populated at begin.
    double sampleRate{16000.0};
    double chunkSizeSec{0.5};
    int32_t unfixedChunkNum{2};
    int32_t unfixedTokenNum{5};
    std::string forceLanguage{};      //!< Empty = no force; e.g. "Chinese".
    std::string context{};            //!< System-prompt context.

    // Per-hop accumulator: precomputed mel frames stored as concatenated
    // fp16 bytes. Each chunk-event mel payload is appended verbatim. The
    // mel tensor shape is [1, mel_bins, T_total]; we track T_total.
    std::vector<uint8_t> melAccumBytes;
    int32_t melBins{128};            //!< Mel bin count; locked from first chunk.
    int32_t melFrames{0};            //!< Cumulative frame count of melAccumBytes.

    // Hop-trigger bookkeeping: number of mel frames at the time of last hop.
    int32_t melFramesAtLastHop{0};

    // Decoded text state.
    std::string rawDecoded{};        //!< Mirrors official state._raw_decoded.
    int32_t chunkId{0};              //!< Hop counter (mirrors official chunk_id).

    // Session-level accumulator for auto-segmentation (Step 4).
    std::string fullText{};
};

//! Convenience wrapper: build the structured kv_capacity_exceeded error event
//! the design doc §12 milestone 2 calls for. M3 routes through this when the
//! runtime returns false with status kKvCapacityExceeded.
Json makeKvCapacityErrorEvent(std::string const& id, int32_t kvLength, int32_t cap)
{
    Json ev = {
        {"event", "error"},
        {"ok", false},
        {"error", "kv_capacity_exceeded"},
        {"kv_length", kvLength},
        {"cap", cap},
    };
    if (!id.empty())
    {
        ev["id"] = id;
    }
    return ev;
}

//! Maps a runtime AppendPrefillStatus to the structured worker-side JSON event.
std::optional<Json> mapAppendStatusToErrorEvent(
    rt::LLMInferenceSpecDecodeRuntime const& runtime, std::string const& id)
{
    using Status = rt::LLMInferenceSpecDecodeRuntime::AppendPrefillStatus;
    auto const status = runtime.getLastAppendStatus();
    switch (status)
    {
    case Status::kOk: return std::nullopt;
    case Status::kKvCapacityExceeded:
        return makeKvCapacityErrorEvent(id, runtime.getLastObservedKvLength(), runtime.getMaxKvCacheCapacity());
    case Status::kChunkTooLong:
    case Status::kPreconditionFailed:
    case Status::kPrefillFailed:
    default: return std::nullopt;
    }
}

enum OptionId : int
{
    HELP = 1000,
    ENGINE_DIR,
    MULTIMODAL_ENGINE_DIR,
    DEBUG,
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir=<path> --multimodalEngineDir=<path> [--debug]\n\n"
              << "Reads llm_inference-compatible JSON lines from stdin and writes JSON lines to stdout.\n";
}

bool parseArgs(Args& args, int argc, char** argv)
{
    static struct option options[] = {{"help", no_argument, 0, HELP},
        {"engineDir", required_argument, 0, ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, MULTIMODAL_ENGINE_DIR}, {"debug", no_argument, 0, DEBUG},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case HELP: printUsage(argv[0]); std::exit(EXIT_SUCCESS);
        case ENGINE_DIR: args.engineDir = optarg; break;
        case MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case DEBUG: args.debug = true; break;
        default: return false;
        }
    }

    return !args.engineDir.empty() && !args.multimodalEngineDir.empty();
}

std::filesystem::path writeTempInput(Json const& input, std::string const& id)
{
    std::string safeId = id.empty() ? "request" : id;
    for (auto& ch : safeId)
    {
        bool const ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
            || ch == '-';
        if (!ok)
        {
            ch = '_';
        }
    }
    auto path = std::filesystem::temp_directory_path()
        / ("qwen3_asr_worker_" + safeId + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
            + ".json");
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open temp input file: " + path.string());
    }
    file << input.dump();
    return path;
}

// ---------------------------------------------------------------------------
// SPIKE — replaced in step 3.
// Step 2 chunk handler: per-chunk full one-shot decode via existing
// handleRequest path. No prefix prompt. The test driver writes per-hop
// mel safetensors files containing the FULL audio accumulated so far
// (hop k = first 500*(k+1) ms). Worker just builds the legacy one-shot
// request JSON pointing at that mel and times the call.
// Step 3 will replace this with prefix-prompt rollback (§15.6 step 3).
// ---------------------------------------------------------------------------

//! Per-stage timing snapshot pulled from gTimer. Tracks the last-recorded
//! GPU time for each stage of interest. We capture cumulative entry counts
//! between hops so we can diff and isolate this-hop's contribution even
//! when stages report multiple runs per call (e.g. eagle iterations).
struct StageTimingSnapshot
{
    float encoderMs{0.0f};         //!< Sum of new entries for audio_encoder this hop.
    float prefillMs{0.0f};         //!< Sum of new entries for llm_prefill this hop.
    float decodeMs{0.0f};          //!< Sum of new entries for llm_generation this hop.
};

//! Cumulative counters across hops, used to diff per-stage timing slices.
struct StageTimingCounters
{
    size_t encoderEntries{0};
    size_t prefillEntries{0};
    size_t decodeEntries{0};
};

float sumNewEntries(std::string const& stageId, size_t& priorCount)
{
    auto data = gTimer.getTimingData(stageId);
    if (!data.has_value())
    {
        return 0.0f;
    }
    auto const& times = data->gpuTimesMs;
    float sumMs = 0.0f;
    for (size_t i = priorCount; i < times.size(); ++i)
    {
        sumMs += times[i];
    }
    priorCount = times.size();
    return sumMs;
}

StageTimingSnapshot captureStageDelta(StageTimingCounters& counters)
{
    StageTimingSnapshot snap;
    snap.encoderMs = sumNewEntries(metrics::StageNames::kAUDIO_ENCODER, counters.encoderEntries);
    snap.prefillMs = sumNewEntries(metrics::StageNames::kLLM_PREFILL, counters.prefillEntries);
    snap.decodeMs = sumNewEntries(metrics::StageNames::kLLM_GENERATION, counters.decodeEntries);
    return snap;
}

//! Build the legacy one-shot request JSON for a single mel file.
//! Mirrors the request layout in scripts/validate_qwen3_tts_quality_gate.py.
Json buildOneShotRequestForMel(std::string const& melPath, int32_t maxGenerateLength)
{
    Json msg = {
        {"role", "user"},
        {"content", Json::array({Json{{"type", "audio"}, {"audio", melPath}}})},
    };
    Json req = {
        {"messages", Json::array({msg})},
    };
    Json input = {
        {"requests", Json::array({req})},
        {"batch_size", 1},
        {"temperature", 1.0},
        {"top_p", 1.0},
        {"top_k", 1},
        {"max_generate_length", maxGenerateLength},
        {"apply_chat_template", true},
        {"add_generation_prompt", true},
    };
    return input;
}

//! Core: drive one handleRequest pass on a mel file, return (text, timings).
//! Used by handleChunk/handleEnd in spike. Returns std::nullopt on failure.
struct HopResult
{
    bool ok{false};
    std::string text;
    double totalMs{0.0};
    StageTimingSnapshot stages{};
};

HopResult runHop(std::string const& melPath, int32_t maxGenerateLength,
    rt::LLMInferenceSpecDecodeRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap,
    StageTimingCounters& stageCounters)
{
    HopResult result;
    auto const t0 = std::chrono::steady_clock::now();
    std::filesystem::path tempPath;
    try
    {
        Json input = buildOneShotRequestForMel(melPath, maxGenerateLength);
        tempPath = writeTempInput(input, "spike_hop");
        std::vector<rt::LLMGenerationRequest> batched;
        std::tie(loraWeightsMap, batched) = exampleUtils::parseRequestFile(tempPath, -1, -1);
        if (batched.empty())
        {
            throw std::runtime_error("parseRequestFile produced no requests");
        }
        rt::LLMGenerationResponse llmResponse;
        bool const ok = runtime.handleRequest(batched[0], llmResponse, stream);
        result.ok = ok;
        if (ok && !llmResponse.outputTexts.empty())
        {
            result.text = llmResponse.outputTexts[0];
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("runHop exception: %s", e.what());
        result.ok = false;
        result.text = std::string("runHop_exception: ") + e.what();
    }
    if (!tempPath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(tempPath, ec);
    }
    result.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.stages = captureStageDelta(stageCounters);
    return result;
}

void handleBegin(Json const& input, AsrSessionState& session)
{
    std::string const id = input.value("id", "");
    if (session.active)
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "session_already_active"}};
        if (!id.empty())
        {
            ev["id"] = id;
        }
        std::cout << ev.dump() << std::endl;
        return;
    }
    session = AsrSessionState{};
    session.sessionId = id;
    session.active = true;
    session.lastActivity = std::chrono::steady_clock::now();
    session.sampleRate = input.value("sample_rate", 16000.0);
    session.chunkSizeSec = input.value("chunk_size_sec", 0.5);
    session.unfixedChunkNum = input.value("unfixed_chunk_num", 2);
    session.unfixedTokenNum = input.value("unfixed_token_num", 5);
    session.context = input.value("context", std::string{});
    if (input.contains("force_language") && !input["force_language"].is_null())
    {
        session.forceLanguage = input["force_language"].get<std::string>();
    }
    Json ev = {{"event", "begin_ack"}, {"id", id}};
    std::cout << ev.dump() << std::endl;
}

// SPIKE — replaced in step 3.
void handleChunk(Json const& input, AsrSessionState& session,
    rt::LLMInferenceSpecDecodeRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap,
    StageTimingCounters& stageCounters, int32_t maxGenerateLength)
{
    std::string const id = input.value("id", "");
    if (!session.active)
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "no_active_session"}};
        if (!id.empty())
        {
            ev["id"] = id;
        }
        std::cout << ev.dump() << std::endl;
        return;
    }
    if (!input.contains("mel_path") || !input["mel_path"].is_string())
    {
        Json ev = {{"event", "error"}, {"ok", false}, {"error", "chunk_missing_mel_path"}, {"id", id}};
        std::cout << ev.dump() << std::endl;
        return;
    }

    std::string const melPath = input["mel_path"].get<std::string>();
    bool const isLast = input.value("last", false);
    session.lastActivity = std::chrono::steady_clock::now();

    int32_t const hopId = session.chunkId;
    HopResult const hop = runHop(melPath, maxGenerateLength, runtime, stream, loraWeightsMap, stageCounters);
    session.chunkId += 1;
    session.rawDecoded = hop.text;

    Json ev = {
        {"event", isLast ? "final" : "partial"},
        {"id", id},
        {"hop_id", hopId},
        {"ok", hop.ok},
        {"text", hop.text},
        {"elapsed_ms", hop.totalMs},
        {"encoder_ms", hop.stages.encoderMs},
        {"prefill_ms", hop.stages.prefillMs},
        {"decode_ms", hop.stages.decodeMs},
    };
    if (isLast)
    {
        ev["total_ms"] = hop.totalMs;
        std::cout << ev.dump() << std::endl;
        // Free session.
        session = AsrSessionState{};
        return;
    }
    std::cout << ev.dump() << std::endl;
}

// SPIKE — replaced in step 3.
void handleEnd(Json const& /*input*/, AsrSessionState& session,
    rt::LLMInferenceSpecDecodeRuntime& /*runtime*/, cudaStream_t /*stream*/,
    std::unordered_map<std::string, std::string>& /*loraWeightsMap*/,
    StageTimingCounters& /*stageCounters*/, int32_t /*maxGenerateLength*/)
{
    std::string const id = session.sessionId;
    // Spike contract: the driver flags the final hop via last=true on a chunk
    // event. Bare `end` events just close the session.
    session = AsrSessionState{};
    Json ev = {{"event", "end_ack"}, {"id", id}};
    std::cout << ev.dump() << std::endl;
}

// ---------------------------------------------------------------------------
// One-shot legacy handler. Existing M2 behavior preserved verbatim — only
// the surrounding main() loop changes. handleOneShot must produce a JSON
// response byte-equivalent (modulo `total_ms` jitter) to the M2 worker.
// ---------------------------------------------------------------------------
void handleOneShot(Json input, rt::LLMInferenceSpecDecodeRuntime& runtime, cudaStream_t stream,
    std::unordered_map<std::string, std::string>& loraWeightsMap)
{
    Json response;
    std::filesystem::path tempPath;
    auto const requestStart = std::chrono::steady_clock::now();
    try
    {
        std::string const id = input.value("id", "");
        input.erase("id");
        int32_t const batchSizeOverride = input.value("batch_size_override", -1);
        int64_t const maxGenerateLengthOverride = input.value("max_generate_length_override", -1);
        input.erase("batch_size_override");
        input.erase("max_generate_length_override");

        tempPath = writeTempInput(input, id);
        std::vector<rt::LLMGenerationRequest> batchedRequests;
        std::tie(loraWeightsMap, batchedRequests)
            = exampleUtils::parseRequestFile(tempPath, batchSizeOverride, maxGenerateLengthOverride);
        if (batchedRequests.empty())
        {
            throw std::runtime_error("No valid ASR requests found");
        }

        Json responses = Json::array();
        bool ok = true;
        for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
        {
            rt::LLMGenerationResponse llmResponse;
            bool const requestOk = runtime.handleRequest(batchedRequests[requestIdx], llmResponse, stream);
            ok = ok && requestOk;
            for (size_t batchIdx = 0; batchIdx < batchedRequests[requestIdx].requests.size(); ++batchIdx)
            {
                bool const hasOutputText = requestOk && batchIdx < llmResponse.outputTexts.size();
                std::string const text = hasOutputText
                    ? llmResponse.outputTexts[batchIdx]
                    : "TensorRT Edge LLM cannot handle this request. Fails.";
                responses.push_back(Json{
                    {"request_idx", requestIdx}, {"batch_idx", batchIdx}, {"output_text", text}});
            }
        }
        double const totalMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - requestStart).count();
        if (!ok)
        {
            if (auto structuredEv = mapAppendStatusToErrorEvent(runtime, id))
            {
                Json ev = std::move(*structuredEv);
                ev["total_ms"] = totalMs;
                response = std::move(ev);
            }
            else
            {
                response = Json{{"id", id}, {"event", "error"}, {"ok", false}, {"responses", responses},
                    {"total_ms", totalMs}};
            }
        }
        else
        {
            response = Json{{"id", id}, {"event", "done"}, {"ok", true}, {"responses", responses},
                {"total_ms", totalMs}};
        }
    }
    catch (std::exception const& e)
    {
        response = Json{{"event", "error"}, {"ok", false}, {"error", e.what()}};
    }
    if (!tempPath.empty())
    {
        std::error_code ec;
        std::filesystem::remove(tempPath, ec);
    }
    std::cout << response.dump() << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    gLogger.setLevel(args.debug ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    auto pluginHandles = loadEdgellmPluginLib();

    // SPIKE — enable stage timing so the chunk handler can report per-stage ms.
    setProfilingEnabled(true);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto const initStart = std::chrono::steady_clock::now();
    std::unordered_map<std::string, std::string> loraWeightsMap;
    auto runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
        args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
    bool const enableGraph = std::getenv("EDGE_LLM_ASR_CUDA_GRAPH") == nullptr
        || std::string(std::getenv("EDGE_LLM_ASR_CUDA_GRAPH")) != "0";
    if (enableGraph && !runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("CUDA graph capture failed for ASR worker, proceeding without.");
    }
    double const initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - initStart).count();
    std::cout << Json{{"event", "ready"}, {"init_ms", initMs}}.dump() << std::endl;

    // Single-session worker per §15.6. Multi-session is out of scope for P0.
    AsrSessionState session;
    // SPIKE — cumulative stage entry counters; runHop diffs against these.
    StageTimingCounters stageCounters{};
    // SPIKE — per-hop decode budget. Generous default; driver controls hop cadence.
    int32_t const spikeMaxGenerateLength = 200;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }

        Json parsed;
        try
        {
            parsed = Json::parse(line);
        }
        catch (std::exception const& e)
        {
            Json err = {{"event", "error"}, {"ok", false}, {"error", std::string("json_parse_failed: ") + e.what()}};
            std::cout << err.dump() << std::endl;
            continue;
        }

        if (!parsed.contains("event"))
        {
            // Backward-compat one-shot: any line that omits `event` flows through
            // the M2 legacy path. handleRequest behavior unchanged.
            handleOneShot(std::move(parsed), *runtime, stream, loraWeightsMap);
            continue;
        }

        std::string const event = parsed.value("event", "");
        if (event == "begin")
        {
            handleBegin(parsed, session);
        }
        else if (event == "chunk")
        {
            handleChunk(parsed, session, *runtime, stream, loraWeightsMap, stageCounters, spikeMaxGenerateLength);
        }
        else if (event == "end")
        {
            handleEnd(parsed, session, *runtime, stream, loraWeightsMap, stageCounters, spikeMaxGenerateLength);
        }
        else
        {
            Json err = {{"event", "error"}, {"ok", false}, {"error", "unknown_event"}};
            if (parsed.contains("id"))
            {
                err["id"] = parsed["id"];
            }
            std::cout << err.dump() << std::endl;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
