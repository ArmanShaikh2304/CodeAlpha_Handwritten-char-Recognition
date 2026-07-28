const predictButton = document.getElementById("predictBtn");
const clearButton = document.getElementById("clearBtn");
const statusMessage = document.getElementById("statusMessage");
const digitResult = document.getElementById("digitResult");
const confidenceResult = document.getElementById("confidenceResult");
const processedImage = document.getElementById("processedImage");
const probabilityChart = document.getElementById("probabilityChart");
const themeToggle = document.getElementById("themeToggle");

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", isError);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function buildEmptyChart() {
  probabilityChart.innerHTML = "";

  for (let digit = 0; digit <= 9; digit += 1) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${digit}</span>
      <span class="bar-track"><span class="bar-fill" data-digit="${digit}"></span></span>
      <span class="bar-percent" data-percent="${digit}">0.0%</span>
    `;
    probabilityChart.appendChild(row);
  }
}

function updateChart(probabilities) {
  probabilities.forEach((probability, digit) => {
    const bar = probabilityChart.querySelector(`[data-digit="${digit}"]`);
    const percent = probabilityChart.querySelector(`[data-percent="${digit}"]`);

    bar.style.width = formatPercent(probability);
    percent.textContent = formatPercent(probability);
  });
}

function resetResults() {
  digitResult.textContent = "-";
  confidenceResult.textContent = "-";
  processedImage.removeAttribute("src");
  processedImage.classList.remove("visible");
  buildEmptyChart();
  setStatus("");
}

async function predictDigit() {
  if (!window.digitCanvas.hasDrawing()) {
    setStatus("Draw a digit before predicting.", true);
    return;
  }

  predictButton.disabled = true;
  setStatus("Predicting...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: window.digitCanvas.toDataURL(),
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Prediction failed.");
    }

    digitResult.textContent = result.digit;
    confidenceResult.textContent = formatPercent(result.confidence);
    processedImage.src = result.processed_image;
    processedImage.classList.add("visible");
    updateChart(result.probabilities);
    setStatus("Prediction complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    predictButton.disabled = false;
  }
}

function applyTheme(isDark) {
  document.body.classList.toggle("dark", isDark);
  localStorage.setItem("theme", isDark ? "dark" : "light");
}

clearButton.addEventListener("click", () => {
  window.digitCanvas.clear();
  resetResults();
});

predictButton.addEventListener("click", predictDigit);

themeToggle.addEventListener("change", () => {
  applyTheme(themeToggle.checked);
});

const savedTheme = localStorage.getItem("theme");
themeToggle.checked = savedTheme === "dark";
applyTheme(themeToggle.checked);
resetResults();
