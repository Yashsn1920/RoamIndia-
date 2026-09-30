const cities = [
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Kochi", "Goa",
    "Jaipur", "Ahmedabad", "Pune", "Bhopal", "Lucknow", "Patna", "Bhubaneswar", "Guwahati",
    "Srinagar", "Chandigarh", "Indore", "Nagpur", "Varanasi", "Manali", "Shimla", "Darjeeling",
    "Udaipur", "Mysuru", "Amritsar", "Rishikesh", "Haridwar", "Ooty", "Munnar", "Port Blair", "Ladakh"
];
const palette = ["#f15a43", "#2b7966", "#e6a24a", "#4783a0"];
const form = document.getElementById("estimateForm");
const money = value => `₹${Number(value).toLocaleString("en-IN")}`;

function setDefaults() {
    const origin = form.elements.origin;
    const destination = form.elements.destination;
    for (const city of cities) {
        origin.add(new Option(city, city));
        destination.add(new Option(city, city));
    }
    origin.value = "Delhi";
    destination.value = "Goa";
    const today = new Date();
    const departure = new Date(today);
    departure.setDate(departure.getDate() + 30);
    const returning = new Date(departure);
    returning.setDate(returning.getDate() + 7);
    form.elements.departure_date.value = departure.toISOString().slice(0, 10);
    form.elements.return_date.value = returning.toISOString().slice(0, 10);
    form.elements.departure_date.min = today.toISOString().slice(0, 10);
    form.elements.return_date.min = new Date(today.getTime() + 86400000).toISOString().slice(0, 10);
}

function renderEstimate(data) {
    const breakdown = data.breakdown;
    const labels = ["Flights", "Accommodation", "Activities", "Food & local"];
    const keys = ["flights", "hotel", "activities", "food_local"];
    document.getElementById("emptyState").hidden = true;
    document.getElementById("estimateResults").hidden = false;
    document.getElementById("totalMid").textContent = money(data.total_mid);
    document.getElementById("totalLow").textContent = money(data.total_low);
    document.getElementById("totalHigh").textContent = money(data.total_high);
    document.getElementById("totalRange").textContent = `${money(data.total_low)} – ${money(data.total_high)} expected range`;
    document.getElementById("tripDuration").textContent = `${data.meta.trip_days} days`;
    document.getElementById("tripSummary").textContent = `${data.meta.origin} to ${data.meta.destination} · ${data.meta.num_travellers} traveller${data.meta.num_travellers === 1 ? "" : "s"} · ${data.meta.hotel_tier} stay`;
    document.getElementById("costLegend").innerHTML = keys.map((key, index) => `
        <div class="legend-row"><span class="legend-swatch" style="--swatch:${palette[index]}"></span>
        <span>${labels[index]}</span><strong>${money(breakdown[key])}</strong></div>`).join("");

    Plotly.react("costChart", [{
        type: "pie", hole: 0.64, labels, values: keys.map(key => breakdown[key]),
        marker: { colors: palette, line: { color: "#fff", width: 3 } },
        textinfo: "none", sort: false, hovertemplate: "%{label}<br>₹%{value:,.0f}<extra></extra>"
    }], {
        height: 250, margin: { t: 8, r: 8, b: 8, l: 8 }, showlegend: false,
        paper_bgcolor: "transparent", plot_bgcolor: "transparent"
    }, { displayModeBar: false, responsive: true });
}

form.addEventListener("submit", async event => {
    event.preventDefault();
    const error = document.getElementById("formError");
    const submit = form.querySelector("button[type='submit']");
    const payload = Object.fromEntries(new FormData(form));
    error.hidden = true;
    if (payload.origin === payload.destination) {
        error.textContent = "Origin and destination must be different.";
        error.hidden = false;
        return;
    }
    document.getElementById("loadingState").hidden = false;
    submit.disabled = true;
    try {
        const response = await fetch("/api/estimate", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Unable to calculate this trip.");
        renderEstimate(result);
    } catch (exception) {
        error.textContent = exception.message;
        error.hidden = false;
    } finally {
        document.getElementById("loadingState").hidden = true;
        submit.disabled = false;
    }
});

setDefaults();