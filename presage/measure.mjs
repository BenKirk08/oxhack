// runs Presage SmartSpectra on 1 video file + prints what it measured as json
//presage_check.py runs this for you (presage only does node / c++, no python)
//by hand: node --env-file=../.env measure.mjs ../data/1.mov  (key = SMARTSPECTRA_API_KEY)
//the maths runs on this mac, presages server checks the key + sends its pulse model (1st run)
import {
  ProcessingStatus,
  SmartSpectraErrorCode,
  SmartSpectraSDK,
  ValidationCode,
  breathingMetrics,
  cardioMetrics,
  decodeMetrics,
} from "@smartspectra/node-sdk";

const videoPath = process.argv[2];
const frameDelay = Number(process.argv[3] ?? 33); //ms between frames, 33 = ab real time at 30 fps
const apiKey = process.env.SMARTSPECTRA_API_KEY;
if (!videoPath || !apiKey) {
  console.error("needs a video path + SMARTSPECTRA_API_KEY (put it in .env)");
  process.exit(2);
}

const pulse = new Map(); //timestamp -> reading (same reading can come twice)
const breathing = new Map();
const hints = {};  // presages warnings eg "too dark" -> how many frames
const errors = [];
let firstFrame = null;  //1st frames timestamp = time 0 of the video
let lastFrame = null;

//64 bit ints (the timestamps) can come as Long objects -> make them plain numbers
const toNumber = (value) => (typeof value === "number" ? value : Number(value.toString()));

function keep(samples, readings) {
  for (const sample of samples ?? []) {  //?? [] -> nothing if theres no list
    const timestamp = toNumber(sample.timestamp);
    readings.set(timestamp, [timestamp, sample.value, sample.confidence, sample.stable ? 1 : 0]);
  }
}

function read(buffer) {
  const metrics = decodeMetrics(buffer);  //protobuf bytes -> object
  keep(metrics.cardio?.pulseRate, pulse); //?. = skip if missing (like a null check)
  keep(metrics.breathing?.rate, breathing);
}

const sdk = new SmartSpectraSDK({
  apiKey,
  requestedMetrics: [...cardioMetrics, ...breathingMetrics],
  enableAccumulatedOutput: true, // every reading again at the end so none get missed
  enableTelemetry: false,  //dont send presage usage stats
});

const finished = new Promise((resolve) => {
  let running = false;
  sdk.on("processingStatus", (status) => {
    if (status === ProcessingStatus.kRunning) running = true;
    if (status === ProcessingStatus.kError) resolve();
    if (running && status === ProcessingStatus.kIdle) resolve(); //idle after running = end of file
  });
  sdk.on("error", (code, message) => {
    if (code === SmartSpectraErrorCode.kNetworkError) { //1st run downloads presages model
      message += " Presage downloads its model and checks the key online, so try a faster"
        + " connection (e.g. a phone hotspot). Once the model is saved, it needs much less.";
    }
    errors.push({ code, message });
    resolve();  //eg wrong key -> dont wait forever
  });
});
sdk.on("metrics", read);
sdk.on("accumulatedMetrics", read);
sdk.on("frameSentThrough", (sent, timestamp) => {
  firstFrame ??= timestamp; //??= -> only sets it the 1st time
  lastFrame = timestamp;
});
sdk.on("validationStatus", (code, timestamp, hint) => {
  if (code !== ValidationCode.kOk) hints[hint || code] = (hints[hint || code] ?? 0) + 1;
});

try {
  sdk.useFile(videoPath, { interframeDelayMs: frameDelay }); //0 = as fast as it can
  sdk.start();
  await finished;
} catch (error) {
  errors.push({ code: error.code ?? -1, message: error.message });
}
await sdk.destroy();

const origin = firstFrame ?? 0;
const inSeconds = (readings) => [...readings.values()]
  .sort((first, second) => first[0] - second[0])
  .map(([timestamp, ...rest]) => [(timestamp - origin) / 1e6, ...rest]); //µs -> s from the start
const answer = JSON.stringify({
  file: videoPath,
  seconds: lastFrame === null ? 0 : (lastFrame - origin) / 1e6,
  pulse: inSeconds(pulse), //[time s, BPM, confidence 0-100, stable 1/0]
  breathing: inSeconds(breathing),
  hints,
  errors,
});
//exit only once its all written (exiting straight after can cut long output off at 64 KB)
process.stdout.write(answer + "\n", () => process.exit(errors.length ? 1 : 0));
