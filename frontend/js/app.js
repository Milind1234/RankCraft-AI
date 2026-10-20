/**
 * RankCraft-AI Luxury Frontend Application Controller
 * Handles live multi-stage search, tri-view ranking comparisons,
 * NLP query intelligence, what-if CTR sandbox, and metrics modals.
 */

// Global State
let currentSearchResults = null;
let currentActiveView = 'lambdamart';
let currentSelectedProductForSandbox = null;

// Category SVG Icons for rich product visual representations
const CATEGORY_ICONS = {
  "Footwear & Athletics": "👟",
  "Audio & Acoustics": "🎧",
  "Computing & Ergonomics": "⌨️",
  "Office Furniture & Wellness": "🪑",
  "Apparel & Outdoor": "🧥",
  "Fitness Gear & Hydration": "⚡"
};

// DOM Elements
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');
const searchSubmitBtn = document.getElementById('search-submit-btn');
const nlpInspector = document.getElementById('nlp-inspector');
const nlpBadgesContainer = document.getElementById('nlp-badges-container');
const latencyDisplay = document.getElementById('latency-display');
const resultsCountText = document.getElementById('results-count-text');
const gridView = document.getElementById('grid-view');
const splitView = document.getElementById('split-view');
const splitLtrGrid = document.getElementById('split-ltr-grid');
const splitBm25Grid = document.getElementById('split-bm25-grid');

// Modals & Drawer Elements
const drawerOverlay = document.getElementById('drawer-overlay');
const drawerContent = document.getElementById('drawer-content');
const closeDrawerBtn = document.getElementById('close-drawer');

const modalSandbox = document.getElementById('modal-sandbox');
const closeModalSandboxBtn = document.getElementById('close-modal-sandbox');
const modalMetrics = document.getElementById('modal-metrics');
const closeModalMetricsBtn = document.getElementById('close-modal-metrics');
const modalResume = document.getElementById('modal-resume');
const closeModalResumeBtn = document.getElementById('close-modal-resume');

// Nav Buttons
const btnMetrics = document.getElementById('btn-metrics');
const btnSandboxNav = document.getElementById('btn-sandbox-nav');
const btnResume = document.getElementById('btn-resume');
const btnCopyBullets = document.getElementById('btn-copy-bullets');

// Sandbox Sliders
const sbSliderPrice = document.getElementById('sb-slider-price');
const sbSliderDiscount = document.getElementById('sb-slider-discount');
const sbSliderRating = document.getElementById('sb-slider-rating');
const sbSliderReviews = document.getElementById('sb-slider-reviews');
const sbValPrice = document.getElementById('sb-val-price');
const sbValDiscount = document.getElementById('sb-val-discount');
const sbValRating = document.getElementById('sb-val-rating');
const sbValReviews = document.getElementById('sb-val-reviews');
const sbOrigCtr = document.getElementById('sb-orig-ctr');
const sbSimCtr = document.getElementById('sb-sim-ctr');
const sbLiftVal = document.getElementById('sb-lift-val');
const sbProductTitle = document.getElementById('sb-product-title');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  // Perform initial search with default query
  executeSearch(searchInput.value.trim());
  // Pre-fetch metrics for ticker
  fetchSystemMetrics();
});

function setupEventListeners() {
  // Search form submission
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    if (query) executeSearch(query);
  });

  // Query pills
  document.querySelectorAll('.query-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const q = pill.getAttribute('data-query');
      searchInput.value = q;
      executeSearch(q);
    });
  });

  // View tabs
  document.querySelectorAll('.view-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentActiveView = tab.getAttribute('data-view');
      renderResults();
    });
  });

  // Drawer
  closeDrawerBtn.addEventListener('click', closeDrawer);
  drawerOverlay.addEventListener('click', (e) => {
    if (e.target === drawerOverlay) closeDrawer();
  });

  // Modals
  btnMetrics.addEventListener('click', openMetricsModal);
  closeModalMetricsBtn.addEventListener('click', () => modalMetrics.classList.remove('active'));
  modalMetrics.addEventListener('click', (e) => {
    if (e.target === modalMetrics) modalMetrics.classList.remove('active');
  });

  btnSandboxNav.addEventListener('click', () => {
    if (currentSearchResults && currentSearchResults.results.lambdamart.length > 0) {
      openSandbox(currentSearchResults.results.lambdamart[0]);
    } else {
      openSandbox();
    }
  });
  closeModalSandboxBtn.addEventListener('click', () => modalSandbox.classList.remove('active'));
  modalSandbox.addEventListener('click', (e) => {
    if (e.target === modalSandbox) modalSandbox.classList.remove('active');
  });

  btnResume.addEventListener('click', () => modalResume.classList.add('active'));
  closeModalResumeBtn.addEventListener('click', () => modalResume.classList.remove('active'));
  modalResume.addEventListener('click', (e) => {
    if (e.target === modalResume) modalResume.classList.remove('active');
  });

  // Copy bullets
  btnCopyBullets.addEventListener('click', () => {
    const text = document.getElementById('resume-bullets-text').innerText;
    navigator.clipboard.writeText(text).then(() => {
      const orig = btnCopyBullets.innerHTML;
      btnCopyBullets.innerHTML = '<span>✅</span> Copied to Clipboard!';
      setTimeout(() => btnCopyBullets.innerHTML = orig, 2500);
    });
  });

  // Sandbox slider listeners
  const debouncedSimulate = debounce(runSimulation, 120);
  sbSliderPrice.addEventListener('input', () => {
    sbValPrice.textContent = `$${parseFloat(sbSliderPrice.value).toFixed(2)}`;
    debouncedSimulate();
  });
  sbSliderDiscount.addEventListener('input', () => {
    sbValDiscount.textContent = `${sbSliderDiscount.value}%`;
    debouncedSimulate();
  });
  sbSliderRating.addEventListener('input', () => {
    sbValRating.textContent = `${parseFloat(sbSliderRating.value).toFixed(1)} ★`;
    debouncedSimulate();
  });
  sbSliderReviews.addEventListener('input', () => {
    sbValReviews.textContent = parseInt(sbSliderReviews.value).toLocaleString();
    debouncedSimulate();
  });
}

// Fetch System Metrics & Feature Importances
async function fetchSystemMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    if (data.lifts) {
      document.getElementById('ticker-lift').textContent = `+${data.lifts.ndcg_10_lift_pct}%`;
    }
  } catch (err) {
    console.warn('Could not fetch metrics:', err);
  }
}

// Execute Search & Re-Ranking
async function executeSearch(query) {
  if (!query) return;

  searchSubmitBtn.disabled = true;
  searchSubmitBtn.innerHTML = '<span>⚡ Scoring...</span>';

  try {
    const response = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, top_k: 80, limit: 24 })
    });

    if (!response.ok) throw new Error('Search request failed');
    const data = await response.json();
    currentSearchResults = data;

    // Update NLP Query Intelligence Card
    updateNLPInspector(data.query_analysis, data.latency_ms);

    // Render results
    renderResults();

  } catch (err) {
    console.error('Search error:', err);
    gridView.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: #f87171;">
        <h3>Search Error</h3>
        <p>${err.message}</p>
      </div>
    `;
  } finally {
    searchSubmitBtn.disabled = false;
    searchSubmitBtn.innerHTML = '<span>Re-Rank</span>';
  }
}

// Update NLP Inspector UI
function updateNLPInspector(intent, latencyMs) {
  nlpInspector.style.display = 'flex';
  latencyDisplay.textContent = `${latencyMs} ms`;
  document.getElementById('ticker-latency').textContent = `${latencyMs} ms`;

  let badgesHtml = '';

  // Typo corrections
  if (intent.typo_corrections && intent.typo_corrections.length > 0) {
    const fixes = intent.typo_corrections.map(c => `"${c.original}" ➔ "${c.corrected}"`).join(', ');
    badgesHtml += `<span class="badge badge-typo">🔤 Corrected: ${fixes}</span>`;
  }

  // Detected Brand
  if (intent.detected_brand) {
    badgesHtml += `<span class="badge badge-brand">🏷️ Brand: ${intent.detected_brand}</span>`;
  }

  // Detected Category
  if (intent.detected_category) {
    badgesHtml += `<span class="badge badge-cat">📁 Category: ${intent.detected_category}</span>`;
  }

  // Extracted Attributes
  if (intent.detected_attributes && intent.detected_attributes.length > 0) {
    intent.detected_attributes.forEach(attr => {
      badgesHtml += `<span class="badge badge-attr">✨ ${attr}</span>`;
    });
  }

  // Max Price constraint
  if (intent.max_price_constraint) {
    badgesHtml += `<span class="badge badge-price">💰 Price &lt; $${intent.max_price_constraint}</span>`;
  }

  if (!badgesHtml) {
    badgesHtml = `<span class="badge badge-attr">General Intent Detected</span>`;
  }

  nlpBadgesContainer.innerHTML = badgesHtml;
}

// Render Results based on View Selection
function renderResults() {
  if (!currentSearchResults) return;

  const results = currentSearchResults.results;
  resultsCountText.textContent = `Showing 24 of ${currentSearchResults.candidate_count} candidates evaluated`;

  if (currentActiveView === 'split') {
    gridView.style.display = 'none';
    splitView.style.display = 'grid';

    // Render LambdaMART column
    splitLtrGrid.innerHTML = results.lambdamart.slice(0, 12).map((item, idx) => 
      renderProductCard(item, 'lambdamart', idx)
    ).join('');

    // Render BM25 column
    splitBm25Grid.innerHTML = results.bm25.slice(0, 12).map((item, idx) => 
      renderProductCard(item, 'bm25', idx)
    ).join('');

  } else {
    splitView.style.display = 'none';
    gridView.style.display = 'grid';

    const itemsToRender = results[currentActiveView] || results.lambdamart;
    gridView.innerHTML = itemsToRender.map((item, idx) => 
      renderProductCard(item, currentActiveView, idx)
    ).join('');
  }
}

// Render Single Product Card HTML
function renderProductCard(item, viewType, idx) {
  const prod = item.product;
  const isTopRank = idx < 3;
  const categoryIcon = CATEGORY_ICONS[prod.category] || "📦";

  // Rank Badge
  const rankNumber = idx + 1;
  const rankClass = rankNumber === 1 ? 'rank-pill gold' : 'rank-pill';

  // Rank Delta badge (only available in lambdamart view)
  let deltaBadge = '';
  if (viewType === 'lambdamart' && typeof item.rank_delta !== 'undefined') {
    if (item.rank_delta > 0) {
      deltaBadge = `<span class="rank-delta up" title="Promoted ${item.rank_delta} places by LambdaMART over BM25">▲ +${item.rank_delta} spots</span>`;
    } else if (item.rank_delta < 0) {
      deltaBadge = `<span class="rank-delta down" title="Demoted ${Math.abs(item.rank_delta)} places due to lower click affinity">▼ ${item.rank_delta} spots</span>`;
    } else {
      deltaBadge = `<span class="rank-delta neutral">▬ Unchanged</span>`;
    }
  }

  // Feature chips
  const featureChips = (prod.features || []).slice(0, 2).map(f => 
    `<span class="feature-chip">${escapeHtml(f)}</span>`
  ).join('');

  // Discount badge
  const discountPct = Math.round((prod.discount_pct || 0) * 100);
  const discountHtml = discountPct > 0 
    ? `<span class="discount-badge">-${discountPct}%</span>` 
    : '';

  // Metrics Box
  let metricsBoxHtml = '';
  if (viewType === 'lambdamart') {
    const ctr = item.predicted_ctr_pct || 50.0;
    metricsBoxHtml = `
      <div class="card-metrics-box">
        <div class="ctr-meter-label">
          <span>Predicted CTR</span>
          <strong style="color: var(--accent-cyan);">${ctr}%</strong>
        </div>
        <div class="ctr-meter-track">
          <div class="ctr-meter-fill" style="width: ${Math.min(100, ctr)}%;"></div>
        </div>
        <div class="scores-subrow">
          <span>LTR Score: <strong>${item.ltr_score}</strong></span>
          <span>Semantic: <strong>${(item.semantic_score * 100).toFixed(0)}%</strong></span>
        </div>
      </div>
    `;
  } else if (viewType === 'semantic') {
    metricsBoxHtml = `
      <div class="card-metrics-box">
        <div class="ctr-meter-label">
          <span>Dense Semantic Match (MiniLM)</span>
          <strong style="color: var(--accent-purple);">${(item.semantic_score * 100).toFixed(1)}%</strong>
        </div>
        <div class="ctr-meter-track">
          <div class="ctr-meter-fill" style="width: ${(item.semantic_score * 100).toFixed(0)}%; background: linear-gradient(90deg, #8b5cf6, #c084fc);"></div>
        </div>
      </div>
    `;
  } else if (viewType === 'bm25') {
    metricsBoxHtml = `
      <div class="card-metrics-box">
        <div class="ctr-meter-label">
          <span>Lexical BM25 Score</span>
          <strong style="color: #94a3b8;">${item.bm25_score.toFixed(2)}</strong>
        </div>
        <div class="ctr-meter-track">
          <div class="ctr-meter-fill" style="width: ${Math.min(100, item.bm25_score * 5)}%; background: #64748b;"></div>
        </div>
      </div>
    `;
  }

  // Prime shipping pill
  const primePill = prod.is_prime_eligible 
    ? `<span style="font-size: 0.7rem; color: #38bdf8; font-weight: 700;">⚡ Fast Free Shipping</span>` 
    : '';

  // Card HTML
  return `
    <div class="product-card ${isTopRank ? 'top-rank' : ''}" data-id="${prod.id}">
      <div>
        <div class="card-top">
          <span class="${rankClass}">#${rankNumber}</span>
          ${deltaBadge}
        </div>

        <div class="card-body">
          <div class="card-category-brand">
            <span>${categoryIcon}</span>
            <span class="card-brand">${escapeHtml(prod.brand)}</span>
            <span>•</span>
            <span>${escapeHtml(prod.sub_category || prod.category)}</span>
          </div>

          <h3 class="card-title" title="${escapeHtml(prod.title)}">${escapeHtml(prod.title)}</h3>

          <div class="feature-bullets">
            ${featureChips}
          </div>

          <div class="price-row">
            <span class="price-current">$${prod.price.toFixed(2)}</span>
            ${prod.original_price > prod.price ? `<span class="price-original">$${prod.original_price.toFixed(2)}</span>` : ''}
            ${discountHtml}
          </div>

          <div class="social-proof-row">
            <div class="rating-stars">
              <span>★</span>
              <span>${prod.rating.toFixed(1)}</span>
              <span style="color: var(--text-muted); font-weight: 400;">(${prod.review_count.toLocaleString()})</span>
            </div>
            ${primePill}
          </div>

          ${metricsBoxHtml}
        </div>
      </div>

      <div class="card-actions">
        <button class="btn btn-secondary btn-sm" onclick="openFeatureDrawer('${prod.id}')">
          <span>🔍</span> Features
        </button>
        <button class="btn btn-primary btn-sm" onclick="openSandboxFromCard('${prod.id}')">
          <span>⚡</span> Simulate
        </button>
      </div>
    </div>
  `;
}

// Open ML Feature Inspector Drawer
window.openFeatureDrawer = function(productId) {
  if (!currentSearchResults) return;
  const item = currentSearchResults.results.lambdamart.find(r => r.product.id === productId) 
            || currentSearchResults.results.bm25.find(r => r.product.id === productId);
  if (!item) return;

  const prod = item.product;
  const f = item.features || {};

  drawerContent.innerHTML = `
    <div style="margin-bottom: 1.5rem;">
      <span class="badge badge-brand">${escapeHtml(prod.brand)}</span>
      <h4 style="font-family: var(--font-heading); font-size: 1.1rem; color: #fff; margin: 0.5rem 0;">
        ${escapeHtml(prod.title)}
      </h4>
      <div style="font-size: 0.85rem; color: var(--text-secondary);">
        Rank: <strong style="color: var(--accent-purple);">#${item.rank || '--'}</strong> • 
        Price: <strong style="color: #fff;">$${prod.price.toFixed(2)}</strong> • 
        Predicted CTR: <strong style="color: var(--accent-cyan);">${item.predicted_ctr_pct || '--'}%</strong>
      </div>
    </div>

    <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
      Engineered Feature Attribution Signals
    </div>

    <div class="feature-list">
      <div class="feature-item">
        <div>
          <div class="feat-name">Semantic Cosine Score</div>
          <div class="feat-desc">Sentence Transformer (all-MiniLM-L6-v2) similarity</div>
        </div>
        <div class="feat-val">${item.semantic_score ? (item.semantic_score * 100).toFixed(1) + '%' : '--'}</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Lexical BM25 Score</div>
          <div class="feat-desc">Stage 1 token frequency & inverse doc frequency</div>
        </div>
        <div class="feat-val">${item.bm25_score ? item.bm25_score.toFixed(2) : '--'}</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Price Ratio to Category Mean</div>
          <div class="feat-desc">(price - category_avg) / category_avg</div>
        </div>
        <div class="feat-val">${f.price_diff_to_cat_mean ? (f.price_diff_to_cat_mean * 100).toFixed(1) + '%' : '--'}</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Discount Percentage</div>
          <div class="feat-desc">Price reduction off original listing</div>
        </div>
        <div class="feat-val" style="color: var(--accent-green);">${Math.round((prod.discount_pct || 0) * 100)}%</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Smoothed Historical CTR</div>
          <div class="feat-desc">Empirical Bayes click affinity signal</div>
        </div>
        <div class="feat-val">${f.historical_ctr ? (f.historical_ctr * 100).toFixed(2) + '%' : '--'}</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Social Proof Volume</div>
          <div class="feat-desc">log1p(review_count) × rating</div>
        </div>
        <div class="feat-val">${prod.rating.toFixed(1)} ★ (${prod.review_count})</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Jaccard Token Overlap</div>
          <div class="feat-desc">Intersection over union of query & title tokens</div>
        </div>
        <div class="feat-val">${f.jaccard_overlap ? (f.jaccard_overlap * 100).toFixed(1) + '%' : '--'}</div>
      </div>

      <div class="feature-item">
        <div>
          <div class="feat-name">Final LambdaMART Score</div>
          <div class="feat-desc">LightGBM Ranker decision tree ensemble logit</div>
        </div>
        <div class="feat-val" style="color: var(--accent-purple);">${item.ltr_score || '--'}</div>
      </div>
    </div>

    <button class="btn btn-primary" style="width: 100%; margin-top: 2rem; justify-content: center;" onclick="closeDrawer(); openSandboxFromCard('${prod.id}');">
      <span>⚡</span> Open in What-If Sandbox
    </button>
  `;

  drawerOverlay.classList.add('active');
};

function closeDrawer() {
  drawerOverlay.classList.remove('active');
}

// Open Sandbox Modal from Product Card
window.openSandboxFromCard = function(productId) {
  if (!currentSearchResults) return;
  const item = currentSearchResults.results.lambdamart.find(r => r.product.id === productId)
            || currentSearchResults.results.bm25.find(r => r.product.id === productId);
  if (item) openSandbox(item);
};

function openSandbox(item) {
  if (!item && currentSearchResults && currentSearchResults.results.lambdamart.length > 0) {
    item = currentSearchResults.results.lambdamart[0];
  }
  if (!item) return;

  currentSelectedProductForSandbox = item;
  const prod = item.product;

  sbProductTitle.textContent = prod.title;
  sbSliderPrice.value = Math.round(prod.price);
  sbValPrice.textContent = `$${prod.price.toFixed(2)}`;

  sbSliderDiscount.value = Math.round((prod.discount_pct || 0) * 100);
  sbValDiscount.textContent = `${sbSliderDiscount.value}%`;

  sbSliderRating.value = prod.rating;
  sbValRating.textContent = `${prod.rating.toFixed(1)} ★`;

  sbSliderReviews.value = prod.review_count;
  sbValReviews.textContent = prod.review_count.toLocaleString();

  // Run initial simulation
  runSimulation();
  modalSandbox.classList.add('active');
}

// Run What-If Simulation API
async function runSimulation() {
  if (!currentSelectedProductForSandbox) return;

  const prod = currentSelectedProductForSandbox.product;
  const query = searchInput.value.trim() || 'lightweight running shoes with arch support';

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        product_id: prod.id,
        query: query,
        price_override: parseFloat(sbSliderPrice.value),
        discount_override: parseFloat(sbSliderDiscount.value) / 100.0,
        rating_override: parseFloat(sbSliderRating.value),
        review_count_override: parseInt(sbSliderReviews.value),
        prime_override: true
      })
    });

    if (!res.ok) throw new Error('Simulation failed');
    const data = await res.json();

    sbOrigCtr.textContent = `${data.original.predicted_ctr_pct}%`;
    sbSimCtr.textContent = `${data.simulated.predicted_ctr_pct}%`;

    const lift = data.ctr_lift_pct;
    if (lift >= 0) {
      sbLiftVal.textContent = `+${lift}%`;
      sbLiftVal.style.color = 'var(--accent-green)';
    } else {
      sbLiftVal.textContent = `${lift}%`;
      sbLiftVal.style.color = 'var(--accent-rose)';
    }
  } catch (err) {
    console.error('Simulation error:', err);
  }
}

// Open Metrics & Feature Importance Modal
async function openMetricsModal() {
  modalMetrics.classList.add('active');
  const container = document.getElementById('feature-importance-bars');
  container.innerHTML = '<div style="color: var(--text-muted);">Loading feature importances...</div>';

  try {
    const res = await fetch('/api/features/importance');
    if (!res.ok) throw new Error('Could not fetch importance');
    const feats = await res.json();

    let html = '';
    feats.slice(0, 10).forEach(f => {
      const pct = (f.importance_gain * 100).toFixed(1);
      html += `
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.25rem;">
            <span>${formatFeatureName(f.feature)}</span>
            <strong style="color: var(--accent-purple);">${pct}%</strong>
          </div>
          <div style="height: 6px; background: rgba(30, 41, 59, 0.8); border-radius: 999px; overflow: hidden;">
            <div style="height: 100%; width: ${pct}%; background: linear-gradient(90deg, #8b5cf6, #06b6d4); border-radius: 999px;"></div>
          </div>
        </div>
      `;
    });
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = '<div style="color: #f87171;">Failed to load feature importance.</div>';
  }
}

// Helpers
function formatFeatureName(raw) {
  return raw.replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
