const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');
const dashboard = document.getElementById('dashboard');

let currentPlayerId = null;

// Currency formatter
const formatMoney = (amount) => {
    if (amount >= 1000000) {
        return '€' + (amount / 1000000).toFixed(2) + 'm';
    } else if (amount >= 1000) {
        return '€' + (amount / 1000).toFixed(0) + 'k';
    }
    return '€' + amount;
};

// Debounce helper
const debounce = (func, wait) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
};

// Handle Search
searchInput.addEventListener('input', debounce(async (e) => {
    const query = e.target.value.trim();
    if (query.length < 3) {
        searchResults.classList.add('hidden');
        return;
    }

    try {
        const res = await fetch(`/api/search?q=${query}`);
        const data = await res.json();
        
        searchResults.innerHTML = '';
        if (data.results.length === 0) {
            searchResults.innerHTML = '<div class="result-item"><div class="result-info"><p>No players found...</p></div></div>';
        } else {
            data.results.forEach(p => {
                const div = document.createElement('div');
                div.className = 'result-item';
                div.innerHTML = `
                    <img src="${p.image_url}" class="result-img" onerror="this.src='https://ui-avatars.com/api/?name=${p.name}&background=random'">
                    <div class="result-info">
                        <h4>${p.name}</h4>
                        <p>${p.club} • ${p.position} • ${formatMoney(p.actual_value)}</p>
                    </div>
                `;
                div.onclick = () => selectPlayer(p.id);
                searchResults.appendChild(div);
            });
        }
        searchResults.classList.remove('hidden');
    } catch (err) {
        console.error("Search failed", err);
    }
}, 300));

// Load Player Profile
async function selectPlayer(id) {
    searchResults.classList.add('hidden');
    searchInput.value = '';
    
    try {
        const res = await fetch(`/api/player/${id}`);
        const data = await res.json();
        const stats = data.stats;
        
        currentPlayerId = id;

        // UI Updates
        document.getElementById('playerName').innerText = stats.name;
        document.getElementById('playerClub').innerText = stats.club;
        document.getElementById('playerPosition').innerText = stats.position;
        document.getElementById('actualValue').innerText = formatMoney(stats.actual_value);
        document.getElementById('aiValue').innerText = formatMoney(data.predicted_value);
        
        const img = document.getElementById('playerImage');
        img.src = stats.image_url;
        img.onerror = () => { img.src = `https://ui-avatars.com/api/?name=${stats.name}&background=random`; };

        // Set baseline sliders
        document.getElementById('sim-age').value = stats.age;
        document.getElementById('val-age').innerText = stats.age;

        document.getElementById('sim-recent_goals').value = stats.recent_goals;
        document.getElementById('val-recent_goals').innerText = stats.recent_goals;

        document.getElementById('sim-recent_assists').value = stats.recent_assists;
        document.getElementById('val-recent_assists').innerText = stats.recent_assists;

        document.getElementById('sim-contract').value = stats.contract_years;
        document.getElementById('val-contract').innerText = stats.contract_years;

        document.getElementById('sim-minutes').value = Math.min(stats.career_minutes, 60000);
        document.getElementById('val-minutes').innerText = stats.career_minutes;

        dashboard.classList.remove('hidden');
        dashboard.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error("Failed to load player", err);
    }
}

// Handle Sliders Change (What-If Simulator)
const sliders = ['age', 'recent_goals', 'recent_assists', 'contract', 'minutes'];

const triggerPrediction = debounce(async () => {
    if (!currentPlayerId) return;

    // Collect overrides
    const overrides = {
        'age_at_valuation': parseFloat(document.getElementById('sim-age').value),
        'recent_goals': parseFloat(document.getElementById('sim-recent_goals').value),
        'recent_assists': parseFloat(document.getElementById('sim-recent_assists').value),
        'years_left_on_contract_at_valuation': parseFloat(document.getElementById('sim-contract').value),
        'career_minutes': parseFloat(document.getElementById('sim-minutes').value)
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_id: currentPlayerId, overrides: overrides })
        });
        const data = await res.json();
        
        // Animate value change
        const aeElem = document.getElementById('aiValue');
        aeElem.style.transform = 'scale(1.1)';
        aeElem.style.color = '#fff';
        setTimeout(() => {
            aeElem.innerText = formatMoney(data.predicted_value);
            aeElem.style.transform = 'scale(1)';
            aeElem.style.color = 'var(--success)';
        }, 150);

    } catch (err) {
        console.error("Prediction API failed", err);
    }
}, 400);

sliders.forEach(key => {
    const input = document.getElementById(`sim-${key}`);
    const label = document.getElementById(`val-${key}`);
    
    input.addEventListener('input', (e) => {
        label.innerText = e.target.value;
        triggerPrediction();
    });
});

// Close search when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box')) {
        searchResults.classList.add('hidden');
    }
});
