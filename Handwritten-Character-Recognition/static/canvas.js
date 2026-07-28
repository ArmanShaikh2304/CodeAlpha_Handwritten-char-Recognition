const canvas = document.getElementById("drawingCanvas");
const context = canvas.getContext("2d");

let isDrawing = false;
let hasDrawing = false;

function fillCanvas() {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function getPointerPosition(event) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function startDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  hasDrawing = true;
  canvas.setPointerCapture(event.pointerId);

  const position = getPointerPosition(event);
  context.beginPath();
  context.moveTo(position.x, position.y);
  context.lineTo(position.x, position.y);
  context.stroke();
}

function draw(event) {
  if (!isDrawing) {
    return;
  }

  event.preventDefault();
  const position = getPointerPosition(event);

  context.lineTo(position.x, position.y);
  context.stroke();
}

function stopDrawing() {
  if (!isDrawing) {
    return;
  }

  isDrawing = false;
  context.closePath();
}

function clearCanvas() {
  fillCanvas();
  hasDrawing = false;
}

context.lineWidth = 22;
context.lineCap = "round";
context.lineJoin = "round";
context.strokeStyle = "#ffffff";
fillCanvas();

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

window.digitCanvas = {
  clear: clearCanvas,
  hasDrawing: () => hasDrawing,
  toDataURL: () => canvas.toDataURL("image/png"),
};
