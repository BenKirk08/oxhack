// runs in the browser when the camera box appears -> app.py gives it to gr.HTML as js_on_load
//gets the newest camera picture from the server, draws it then asks for the next one forever
//so it plays like a video. "element" = the camera box (gradio gives us it)
const canvas = element.querySelector("canvas");
const context = canvas.getContext("2d"); //the canvas drawing tools
let shown = -1;  // number of last frame shown (-1 = none yet)

const sleep = (milliseconds) => new Promise((wake) => setTimeout(wake, milliseconds));

async function showFrames() {
  while (true) {
    try {
      const response = await fetch(`/camera.jpg?after=${shown}`, { cache: "no-store" });
      if (response.status !== 200) { //no frame yet (camera starting) -> try again soon
        await sleep(200);
        continue;
      }
      shown = Number(response.headers.get("X-Frame-Number"));
      const frame = await createImageBitmap(await response.blob());  // decode the jpeg
      if (canvas.width !== frame.width || canvas.height !== frame.height) {
        canvas.width = frame.width; //match the cameras size (page scales it after)
        canvas.height = frame.height;
      }
      context.drawImage(frame, 0, 0);
      frame.close(); // free it now or safaris memory piles up (ab 1 GB!)
    } catch (error) {  //server stopped / restarting -> wait then try agian
      await sleep(1000);
    }
  }
}

showFrames();
