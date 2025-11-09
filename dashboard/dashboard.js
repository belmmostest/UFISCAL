/* global Chart */

(function () {
  // ---------- Helpers ----------------------------------------------------
  const $ = (sel) => document.querySelector(sel);

  const fmt = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 1,
  });

  function millions(val) {
    return fmt.format(val);
  }

  function setJSON(element, obj) {
    element.textContent = JSON.stringify(obj, null, 2);
  }

  // ---------- Quick-simulation tab --------------------------------------
  const form = $("#quickForm");
  const jsonEl = $("#rawJSON");
  const gdpCtx = $("#gdpChart");
  const revCtx = $("#revPie");

  let gdpChart, revChart;

  async function runQuickSim(payload) {
    jsonEl.textContent = "Running …";

    const res = await fetch("/api/simulate/quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`API error ${res.status}`);

    const data = await res.json();
    setJSON(jsonEl, data);

    // ---- GDP line chart -----
    if (data.dynamic_path) {
      const years = data.dynamic_path.year_index;
      const gdp = data.dynamic_path.gdp;

      if (gdpChart) gdpChart.destroy();
      gdpChart = new Chart(gdpCtx, {
        type: "line",
        data: {
          labels: years,
          datasets: [
            {
              label: "GDP (millions AED)",
              data: gdp,
              borderColor: "#0d6efd",
              tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false },
          },
        },
      });
    }

    // ---- Revenue pie -----
    if (data.revenue_analysis) {
      const r = data.revenue_analysis;
      const labels = ["CIT", "VAT", "Oil royalties"];
      const values = [r.corporate_component, r.vat_component, r.oil_component];

      if (revChart) revChart.destroy();
      revChart = new Chart(revCtx, {
        type: "pie",
        data: {
          labels,
          datasets: [
            {
              data: values,
              backgroundColor: ["#0d6efd", "#6f42c1", "#198754"],
            },
          ],
        },
        options: {
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });
    }
  }

  // ---------- Form submit ----------------------------------------------
  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const fd = new FormData(form);
    const payload = {
      standard_rate: (+fd.get("standard_rate") || 0) / 100,
      vat_rate: (+fd.get("vat_rate") || 0) / 100,
      compliance_rate: (+fd.get("compliance_rate") || 0) / 100,
      government_spending_rel_change:
        (+fd.get("government_spending_rel_change") || 0) / 100,
      small_biz_threshold: +fd.get("small_biz_threshold"),
      threshold: +fd.get("threshold"),
      years: +fd.get("years") || 5,
    };

    runQuickSim(payload).catch((err) => {
      jsonEl.textContent = `Error: ${err.message}`;
    });
  });
})();
