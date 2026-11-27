"use strict";

const sidebar = document.getElementById("sidebar");
const menuButton = document.getElementById("mobileMenu");
const backdrop = document.getElementById("sidebarBackdrop");
const mobileViewport = window.matchMedia("(max-width: 900px)");

function setMenuOpen(isOpen, returnFocus = false) {
    const shouldOpen = isOpen && mobileViewport.matches;

    sidebar.classList.toggle("open", shouldOpen);
    backdrop.classList.toggle("is-visible", shouldOpen);
    menuButton.setAttribute("aria-expanded", String(shouldOpen));
    menuButton.setAttribute(
        "aria-label",
        shouldOpen ? "Close navigation menu" : "Open navigation menu"
    );
    document.body.classList.toggle("nav-open", shouldOpen);

    if (shouldOpen) {
        sidebar.querySelector("a")?.focus();
    } else if (returnFocus) {
        menuButton.focus();
    }
}

menuButton.addEventListener("click", () => {
    setMenuOpen(menuButton.getAttribute("aria-expanded") !== "true");
});
backdrop.addEventListener("click", () => setMenuOpen(false));

sidebar.querySelectorAll("nav a").forEach((link) => {
    link.addEventListener("click", () => {
        sidebar.querySelectorAll("nav a").forEach((item) => {
            item.classList.remove("active");
            item.removeAttribute("aria-current");
        });

        link.classList.add("active");
        link.setAttribute("aria-current", "location");
        setMenuOpen(false);
    });
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
        setMenuOpen(false, true);
    }
});

mobileViewport.addEventListener("change", (event) => {
    if (!event.matches) {
        setMenuOpen(false);
    }
});

const forecastForm = document.getElementById("forecastForm");
const citySelect = document.getElementById("citySelect");
const wardSelect = document.getElementById("wardSelect");
const generateButton = document.getElementById("generateForecast");
const buttonLabel = document.getElementById("buttonLabel");
const buttonSpinner = document.getElementById("buttonSpinner");
const forecastResults = document.getElementById("forecast-results");
const forecastMessage = document.getElementById("forecastMessage");
const resultBadge = document.getElementById("resultBadge");
const forecastEmpty = document.getElementById("forecastEmpty");
const forecastLoading = document.getElementById("forecastLoading");
const forecastError = document.getElementById("forecastError");
const forecastErrorMessage = document.getElementById("forecastErrorMessage");
const forecastSuccess = document.getElementById("forecastSuccess");
const driversEmpty = document.getElementById("driversEmpty");
const driverList = document.getElementById("driverList");
const chartEmpty = document.getElementById("chartEmpty");
const chartContent = document.getElementById("chartContent");
const asOfDate = document.getElementById("asOfDate");
const currentMode = document.getElementById("currentMode");
let requestSequence = 0;
let activeForecastController = null;
const FORECAST_TIMEOUT_MS = 30000;

function availableHorizons() {
    try {
        return JSON.parse(forecastForm.dataset.availableHorizons || "[]").map(Number);
    } catch (_) {
        return [];
    }
}

function dateBounds() {
    try {
        return JSON.parse(forecastForm.dataset.dateBounds || "{}");
    } catch (_) {
        return {};
    }
}

function selectedHorizon() {
    return Number(forecastForm.querySelector('input[name="horizon"]:checked')?.value);
}

function updateDateBounds() {
    const allBounds = dateBounds();
    const areaBounds = allBounds.areas?.[citySelect.value]?.[wardSelect.value];
    const bounds = areaBounds || { min: allBounds.min, max: allBounds.max };
    if (!bounds.min || !bounds.max) return;
    asOfDate.min = bounds.min;
    asOfDate.max = bounds.max;
    if (!asOfDate.value || asOfDate.value < bounds.min || asOfDate.value > bounds.max) {
        asOfDate.value = bounds.max;
    }
}

function refreshControlAvailability() {
    const available = availableHorizons();
    forecastForm.querySelectorAll('input[name="horizon"]').forEach((input) => {
        input.disabled = !available.includes(Number(input.value));
    });
    citySelect.disabled = !citySelect.options.length || citySelect.options.length <= 1;
    wardSelect.disabled = !citySelect.value || wardSelect.options.length <= 1;
    currentMode.disabled = currentMode.dataset.available !== "true";
    asOfDate.disabled = currentMode.checked || available.length === 0;
    asOfDate.required = !currentMode.checked;
    generateButton.disabled = forecastResults.getAttribute("aria-busy") === "true"
        || available.length === 0 || !citySelect.value || !wardSelect.value;
}

function populateWards() {
    const cityOption = citySelect.selectedOptions[0];
    let wards = [];
    try {
        wards = cityOption?.dataset.wards ? JSON.parse(cityOption.dataset.wards) : [];
    } catch (_) {
        wards = [];
    }

    wardSelect.replaceChildren(new Option(wards.length ? "Select a ward" : "Select a city first", ""));
    wardSelect.disabled = wards.length === 0;

    wards.forEach((wardId) => {
        const label = `Ward ${wardId.split("_Ward_").at(-1)}`;
        wardSelect.add(new Option(label, wardId));
    });
}

function setForecastState(state, message = "") {
    const isLoading = state === "loading";
    const horizonInputs = forecastForm.querySelectorAll('input[name="horizon"]');

    forecastResults.setAttribute("aria-busy", String(isLoading));
    forecastEmpty.hidden = state !== "empty";
    forecastLoading.hidden = !isLoading;
    forecastError.hidden = state !== "error";
    forecastSuccess.hidden = state !== "success";
    driversEmpty.hidden = state === "success";
    driverList.hidden = state !== "success";
    chartEmpty.hidden = state === "success";
    chartContent.hidden = state !== "success";
    refreshControlAvailability();
    buttonSpinner.hidden = !isLoading;
    buttonLabel.textContent = isLoading ? "Preparing forecast" : "Generate forecast";

    if (state === "loading") {
        resultBadge.textContent = "Preparing";
        forecastMessage.textContent = "Preparing your selected forecast.";
    } else if (state === "error") {
        resultBadge.textContent = "Unavailable";
        forecastMessage.textContent = "No prediction was generated.";
        forecastErrorMessage.textContent = message;
    } else if (state === "success") {
        resultBadge.textContent = "Forecast ready";
        forecastMessage.textContent = "Forecast results are ready.";
    } else {
        resultBadge.textContent = "Awaiting forecast";
        forecastMessage.textContent = "Select a city and ward to prepare your forecast request.";
        forecastErrorMessage.textContent = "";
    }
}

function clearForecast() {
    setForecastState("empty");
}

function invalidatePendingForecast() {
    requestSequence += 1;
    if (activeForecastController) {
        activeForecastController.abort();
        activeForecastController = null;
    }
}

citySelect.addEventListener("change", () => {
    invalidatePendingForecast();
    populateWards();
    updateDateBounds();
    clearForecast();
});

wardSelect.addEventListener("change", () => {
    invalidatePendingForecast();
    updateDateBounds();
    clearForecast();
});
forecastForm.querySelectorAll('input[name="horizon"]').forEach((input) => {
    input.addEventListener("change", () => {
        invalidatePendingForecast();
        clearForecast();
    });
});
currentMode.addEventListener("change", () => {
    invalidatePendingForecast();
    refreshControlAvailability();
    clearForecast();
});

function formatDate(value) {
    if (typeof value !== "string" || Number.isNaN(Date.parse(value))) {
        throw new Error("The forecast service returned an invalid observation date.");
    }

    return new Intl.DateTimeFormat("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
    }).format(new Date(value));
}

function validateSeries(series, name, minimumLength) {
    if (
        !Array.isArray(series)
        || series.length < minimumLength
        || series.some((point) => (
            !point
            || typeof point.date !== "string"
            || Number.isNaN(Date.parse(point.date))
            || !Number.isFinite(point.cases)
            || point.cases < 0
        ))
    ) {
        throw new Error(`The forecast service returned invalid ${name} data.`);
    }

    return series;
}

function drawForecastChart(history, forecast, selection) {
    const svgNamespace = "http://www.w3.org/2000/svg";
    const chartGrid = document.getElementById("chartGrid");
    const chartYAxis = document.getElementById("chartYAxis");
    const historyLine = document.getElementById("historyLine");
    const forecastLine = document.getElementById("forecastLine");
    const divider = document.getElementById("forecastDivider");
    const chartDescription = document.getElementById("chartSvgDescription");
    const chartTitle = document.getElementById("chartSvgTitle");
    const allValues = [...history, ...forecast].map((point) => point.cases);
    const maxCases = Math.max(1, ...allValues);
    const plot = { left: 58, right: 736, top: 20, bottom: 236 };
    const totalPoints = history.length + forecast.length;

    chartGrid.replaceChildren();
    chartYAxis.replaceChildren();

    for (let line = 0; line < 5; line += 1) {
        const y = plot.top + ((plot.bottom - plot.top) * line) / 4;
        const gridLine = document.createElementNS(svgNamespace, "line");
        gridLine.setAttribute("x1", String(plot.left));
        gridLine.setAttribute("x2", String(plot.right));
        gridLine.setAttribute("y1", String(y));
        gridLine.setAttribute("y2", String(y));
        chartGrid.append(gridLine);

        const axisLabel = document.createElementNS(svgNamespace, "text");
        axisLabel.setAttribute("x", String(plot.left - 10));
        axisLabel.setAttribute("y", String(y + 4));
        axisLabel.setAttribute("text-anchor", "end");
        axisLabel.textContent = new Intl.NumberFormat(undefined, {
            maximumFractionDigits: 1,
        }).format(maxCases * (4 - line) / 4);
        chartYAxis.append(axisLabel);
    }

    function pointCoordinates(point, index) {
        const x = plot.left + ((plot.right - plot.left) * index) / (totalPoints - 1);
        const y = plot.bottom - ((plot.bottom - plot.top) * point.cases) / maxCases;
        return `${x},${y}`;
    }

    historyLine.setAttribute(
        "points",
        history.map(pointCoordinates).join(" ")
    );
    forecastLine.setAttribute(
        "points",
        [history[history.length - 1], ...forecast]
            .map((point, index) => pointCoordinates(
                point,
                history.length - 1 + index
            ))
            .join(" ")
    );

    const dividerX = plot.left
        + ((plot.right - plot.left) * (history.length - 1)) / (totalPoints - 1);
    divider.setAttribute("x1", String(dividerX));
    divider.setAttribute("x2", String(dividerX));
    divider.setAttribute("y1", String(plot.top));
    divider.setAttribute("y2", String(plot.bottom));

    chartTitle.textContent = `${selection.city} ${wardSelect.selectedOptions[0].textContent}: observed weekly new cases and forecast`;
    chartDescription.textContent = `Observed new dengue cases for each reporting week through ${formatDate(history.at(-1).date)}, followed by a forecast for the target reporting week.`;
    document.getElementById("chartStartDate").textContent = formatDate(history[0].date);
    document.getElementById("chartObservedDate").textContent = formatDate(history.at(-1).date);
    document.getElementById("chartForecastDate").textContent = formatDate(forecast.at(-1).date);
}

function renderForecastResult(result, selection) {
    const expectedPrediction = Number.isFinite(result?.expectedPredictedCases)
        ? result.expectedPredictedCases
        : Math.max(0, result?.rawPredictedCases);
    if (
        !result
        || !Number.isFinite(result.rawPredictedCases)
        || !Number.isFinite(expectedPrediction)
        || expectedPrediction < 0
        || !Number.isFinite(result.predictedCases)
        || result.predictedCases < 0
        || !Number.isInteger(result.predictedCases)
        || typeof result.targetDate !== "string"
        || typeof result.observationDate !== "string"
        || typeof result.riskCategory !== "string"
        || !result.riskCategory.trim()
        || !result.drivers
    ) {
        throw new Error("The forecast service returned an incomplete result.");
    }

    const history = validateSeries(result.history, "historical case", 2);
    const forecast = validateSeries(result.forecast, "forecast", 1);
    if (forecast.length !== 1 || forecast[0].date !== result.targetDate
        || result.horizonWeeks !== selection.horizonWeeks) {
        throw new Error("The forecast response does not match the selected horizon and target week.");
    }
    if (selection.asOf) {
        const requested = Date.parse(`${selection.asOf}T00:00:00Z`);
        const resolved = Date.parse(`${result.observationDate}T00:00:00Z`);
        const target = Date.parse(`${result.targetDate}T00:00:00Z`);
        if (result.requestedAsOf !== selection.asOf || resolved > requested
            || requested - resolved >= 7 * 24 * 60 * 60 * 1000
            || target - resolved !== selection.horizonWeeks * 7 * 24 * 60 * 60 * 1000) {
            throw new Error("The selected calendar date did not resolve to the expected weekly forecast origin.");
        }
    }
    const drivers = [
        ["rainfallDriver", result.drivers.rainfall],
        ["temperatureDriver", result.drivers.temperature],
        ["humidityDriver", result.drivers.humidity],
        ["recentCasesDriver", result.drivers.recentCases],
    ];

    if (drivers.some(([, value]) => typeof value !== "string" || !value.trim())) {
        throw new Error("The forecast service returned incomplete driver explanations.");
    }

    document.getElementById("expectedPredictedCases").textContent =
        new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
            .format(expectedPrediction);
    document.getElementById("roundedPredictedCases").textContent =
        `Rounded case estimate: ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })
            .format(result.predictedCases)} cases`;
    const rawModelOutput = document.getElementById("rawModelOutput");
    rawModelOutput.hidden = !(result.rawPredictedCases < 0 && !result.rawPredictionIsClipped);
    rawModelOutput.textContent = rawModelOutput.hidden
        ? ""
        : `Raw model output before the non-negative case-count constraint: ${new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(result.rawPredictedCases)} cases`;
    document.getElementById("riskCategory").textContent = result.riskCategory;
    document.getElementById("riskCategory").dataset.risk = result.riskCategory;
    document.getElementById("predictionPeriod").textContent =
        `${selection.horizonWeeks}-week horizon · target reporting week: ${formatDate(result.targetDate)}`;
    document.getElementById("predictionModel").textContent =
        `Selected model: ${result.model} · ${selection.horizonWeeks}-week horizon · backend: ${result.predictionBackend || "phase2"}`;
    const forecastKind = result.isHistoricalReplay
        ? "Leakage-safe held-out backtest replay"
        : result.isRetrospectiveEstimate
            ? "Retrospective historical estimate"
            : selection.asOf
                ? "Historical estimate"
                : "Current forecast";
    const requestedDateText = result.requestedAsOf
        ? `requested date: ${formatDate(result.requestedAsOf)} · `
        : "";
    document.getElementById("observationDate").textContent =
        `${forecastKind} · ${selection.city}, ${wardSelect.selectedOptions[0].textContent} · ${requestedDateText}resolved observation: ${formatDate(result.observationDate)} · target week: ${formatDate(result.targetDate)} · model training data through: ${formatDate(result.modelTrainingDataThrough || result.modelTrainingDataCutoff)} · ${result.model}`;
    const riskWarning = document.getElementById("riskWarning");
    riskWarning.textContent = [result.predictionWarning, result.riskWarning, ...(result.warnings || []).filter((warning) => warning !== result.riskWarning && warning !== result.predictionWarning)].filter(Boolean).join(" ")
        || "Experimental ward-relative classifications; not clinically validated public-health alerts.";
    riskWarning.hidden = false;
    if (result.actualCases !== null && Number.isFinite(result.actualCases)) {
        document.getElementById("observationDate").textContent +=
            ` · observed target-week count: ${new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(result.actualCases)}`;
    } else {
        document.getElementById("observationDate").textContent += " · target-week actual is not present in the dataset";
    }

    drivers.forEach(([elementId, value]) => {
        document.getElementById(elementId).textContent = value;
    });

    const featureRows = [
        ["featureAvgTemp", "avg_temp_C", "°C"],
        ["featureHumidity", "relative_humidity_pct", "%"],
        ["featureRainfall", "rainfall_mm", " mm"],
        ["featureRainLag1", "rain_lag_1w", " mm"],
        ["featureRainLag2", "rain_lag_2w", " mm"],
        ["featureRainLag3", "rain_lag_3w", " mm"],
        ["featureRainLag4", "rain_lag_4w", " mm"],
        ["featureRainSum2", "rain_sum_2w", " mm"],
        ["featureRainSum4", "rain_sum_4w", " mm"],
        ["featureDensity", "population_density_sqkm", " people/km²"],
        ["featureWaterlogging", "water_logging_risk_index", ""],
        ["featureSanitation", "sanitation_score", ""],
    ];
    const observedFeatures = result.observedFeatures || {};
    featureRows.forEach(([elementId, featureName, unit]) => {
        const value = observedFeatures[featureName];
        document.getElementById(elementId).textContent = Number.isFinite(value)
            ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value)}${unit}`
            : "Unavailable";
    });
    document.getElementById("observedFeatureList").hidden = false;

    const rainfallTiming = result.rainfallTiming?.target_minus_three_weeks;
    const rainfallTimingSummary = document.getElementById("rainfallTimingSummary");
    rainfallTimingSummary.textContent = rainfallTiming
        ? `Rainfall observed ${formatDate(rainfallTiming.date)} (${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(rainfallTiming.rainfall_mm)} mm) is approximately three weeks before the target reporting week. This is timing context, not evidence of causation.`
        : "Rainfall approximately three weeks before the target week would occur after this forecast origin and is unavailable. No future rainfall observation was used.";
    rainfallTimingSummary.hidden = false;

    renderModelPerformance(result.modelPerformance, selection.horizonWeeks);

    drawForecastChart(history, forecast, selection);
    setForecastState("success");
}

function renderModelPerformance(performance, horizon) {
    const empty = document.getElementById("performanceEmpty");
    const grid = document.getElementById("performanceGrid");
    const note = document.getElementById("performanceNote");
    if (!performance || typeof performance !== "object") {
        empty.hidden = false;
        grid.hidden = true;
        note.hidden = true;
        return;
    }
    const number = (value) => Number.isFinite(value)
        ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)
        : "Unavailable";
    const percent = (value) => Number.isFinite(value)
        ? new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(value)
        : "Unavailable";
    const fields = [
        ["validationMae", performance.validationMae],
        ["validationRmse", performance.validationRmse],
        ["testMae", performance.testMae],
        ["persistenceMae", performance.persistenceMae],
        ["seasonalMae", performance.seasonalNaiveMae],
        ["testR2", performance.testR2],
    ];
    fields.forEach(([id, value]) => { document.getElementById(id).textContent = number(value); });
    const risk = performance.risk || {};
    document.getElementById("riskExact").textContent = percent(risk.exact_tier_accuracy);
    document.getElementById("riskRecall").textContent = percent(risk.high_critical_recall);
    document.getElementById("criticalRecall").textContent = percent(risk.critical_recall);
    empty.hidden = true;
    grid.hidden = false;
    note.hidden = false;
    note.textContent = `${performance.model} · ${horizon}-week model · validation MAE ${number(performance.validationMae)} · test origins ${formatDate(performance.originStart)} to ${formatDate(performance.originEnd)}; targets ${formatDate(performance.targetStart)} to ${formatDate(performance.targetEnd)}. Historical rolling-origin estimates only.`;
}

async function requestForecast(selection, signal) {
    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
    const response = await fetch("/api/predict", {
        method: "POST",
        credentials: "same-origin",
        signal,
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({
            city: selection.city,
            ward: selection.ward,
            horizonWeeks: selection.horizonWeeks,
            asOf: selection.asOf || null,
        }),
    });
    let data = {};
    try {
        data = await response.json();
    } catch (_) {
        // A non-JSON server response is rendered as a generic service error.
    }
    if (!response.ok) {
        if (response.status === 403
            && data.detail === "Invalid or missing CSRF token.") {
            throw new Error(
                "Your dashboard session expired or changed. Refresh this page, sign in again if prompted, then submit the forecast again."
            );
        }
        const detail = typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
                ? data.detail.map((item) => item.msg || "Invalid request").join("; ")
                : `Forecast request failed (${response.status}).`;
        throw new Error(detail);
    }
    return data;
}

forecastForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!forecastForm.reportValidity()
        || !citySelect.value || !wardSelect.value
        || !availableHorizons().includes(selectedHorizon())) {
        return;
    }

    const selection = {
        city: citySelect.value,
        ward: wardSelect.value,
        horizonWeeks: Number(
            forecastForm.querySelector('input[name="horizon"]:checked').value
        ),
        asOf: currentMode.checked ? null : asOfDate.value,
    };

    if (activeForecastController) activeForecastController.abort();
    const controller = new AbortController();
    activeForecastController = controller;
    const sequence = ++requestSequence;
    const timeout = window.setTimeout(() => controller.abort(), FORECAST_TIMEOUT_MS);
    setForecastState("loading");

    try {
        const result = await requestForecast(selection, controller.signal);
        if (sequence !== requestSequence) return;
        renderForecastResult(result, selection);
    } catch (error) {
        if (sequence !== requestSequence) return;
        const message = controller.signal.aborted
            ? "The forecast request timed out after 30 seconds. Your division and ward selections are available; retry the forecast."
            : error instanceof Error
                ? error.message
                : "The forecast could not be generated. No prediction was returned.";
        setForecastState("error", message);
    } finally {
        window.clearTimeout(timeout);
        if (activeForecastController === controller) activeForecastController = null;
    }
});

currentMode.dataset.available = currentMode.disabled ? "false" : "true";
if (!forecastForm.querySelector('input[name="horizon"]:checked')
    || !availableHorizons().includes(selectedHorizon())) {
    const firstAvailable = forecastForm.querySelector(`input[name="horizon"][value="${availableHorizons()[0]}"]`);
    if (firstAvailable) firstAvailable.checked = true;
}
populateWards();
updateDateBounds();
refreshControlAvailability();
