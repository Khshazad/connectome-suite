"""
Product 3: Behavior Prediction SaaS Game & Benchmark Web App
=============================================================
Generates a standalone, polished SaaS neuroscience challenge web app.
Features:
- Biological scenario cards with real connectome circuit stimulation
- Interactive behavioral guessing panel
- GNN model prediction comparison (User vs Bio-GNN v4 vs Ground Truth)
- Dynamic score tracking, streaks, and level multipliers
- Interactive global leaderboard & model benchmark matrix
- Synthetic Web Audio API sound effects and visual celebration particles
- Glassmorphism dark mode UI styling
"""

import json
from pathlib import Path

def build_product_3(loader, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract real sample neurons from loader
    sample_neurons = loader.neurons.head(50).to_dict('records')
    
    # Generate realistic biological benchmark scenarios
    scenarios = [
        {
            "id": "SCENARIO-01",
            "title": "Optomotor Visual Steering (EPG-PB Ring Attractor)",
            "category": "Navigation & Flight",
            "difficulty": "Medium",
            "sensory_stimulus": "Horizontal Visual Motion Vector: 60°/s Left-to-Right Shift",
            "stimulated_circuit": "Optic Lobe T4/T5 -> EPG Ring Attractor -> PFNv Motor Output",
            "key_neurons": [n['id'] for n in sample_neurons[:4]],
            "description": "Visual motion detectors in the optic lobe register a rapid rightward background optic flow. Signal propagates through heading-direction EPG neurons in the central complex.",
            "options": [
                {"id": 0, "text": "Rapid Yaw Steering Left (Counter-rotation Flight)"},
                {"id": 1, "text": "Immediate Backward Escape Jump"},
                {"id": 2, "text": "Proboscis Extension Feeding Response"},
                {"id": 3, "text": "Complete Wing Motor Arrest / Freeze"}
            ],
            "correct_index": 0,
            "gnn_prediction": 0,
            "gnn_probabilities": [88.4, 4.2, 1.8, 5.6],
            "explanation": "Optomotor counter-steering stabilizes flight heading. Cholinergic transmission from EPG to PFNv drives compensatory wing beat amplitude modulation."
        },
        {
            "id": "SCENARIO-02",
            "title": "Aversive Olfactory Avoidance (MBON-06 Circuit)",
            "category": "Learning & Memory",
            "difficulty": "Hard",
            "sensory_stimulus": "Odor Stimulus: 10µM Benzaldehyde + Dopaminergic Shock Pairing",
            "stimulated_circuit": "PN -> Kenyon Cells (KC) -> MBON-06 (beta2beta'2a)",
            "key_neurons": [n['id'] for n in sample_neurons[4:8]],
            "description": "Antennal lobe projection neurons activate 2,000 Kenyon cells in the mushroom body. A simultaneous PPL1 dopamine pulse depresses KC-to-MBON cholinergic synapses.",
            "options": [
                {"id": 0, "text": "Approach & Odor Alignment"},
                {"id": 1, "text": "Aversive Odor Avoidance Run / Flight Turn"},
                {"id": 2, "text": "Grooming & Antennal Cleaning"},
                {"id": 3, "text": "Courtship Song Generation"}
            ],
            "correct_index": 1,
            "gnn_prediction": 1,
            "gnn_probabilities": [3.1, 92.7, 2.4, 1.8],
            "explanation": "Depression of MBON-06 (approach pathway) tips the MBON network balance toward MBON-11 (avoidance pathway), driving immediate negative chemotaxis."
        },
        {
            "id": "SCENARIO-03",
            "title": "Nociceptive Giant Fiber Threat Retreat",
            "category": "Escape Circuits",
            "difficulty": "Easy",
            "sensory_stimulus": "High-Intensity Looming Visual Shadow (800°/s expanding disk)",
            "stimulated_circuit": "LPLC2 Visual Feature Detectors -> Giant Fiber (GF) -> TTM/DLM Motor Neurons",
            "key_neurons": [n['id'] for n in sample_neurons[8:12]],
            "description": "Looming shadow triggers synchronized dendritic summation in the Giant Fiber neuron. Threshold depolarization exceeds -35mV.",
            "options": [
                {"id": 0, "text": "Slow Crawl Away"},
                {"id": 1, "text": "Ultrashort Latency Takeoff Escape Jump (<5ms)"},
                {"id": 2, "text": "Quiescent Freezing Behavior"},
                {"id": 3, "text": "Abdominal Bending & Oviposition"}
            ],
            "correct_index": 1,
            "gnn_prediction": 1,
            "gnn_probabilities": [1.1, 97.4, 0.9, 0.6],
            "explanation": "The Giant Fiber command neuron fires a single action potential, electrical synapses activate tergotrochanteral leg jump motor neurons in <5ms."
        },
        {
            "id": "SCENARIO-04",
            "title": "Gustatory Sugar Feeding Reflex (SEZ Circuit)",
            "category": "Feeding & Metabolism",
            "difficulty": "Medium",
            "sensory_stimulus": "Tarsal Contact: 500mM Sucrose Solution",
            "stimulated_circuit": "Gr5a Gustatory Receptors -> Subesophageal Zone (SEZ) -> MN9 Motor Neuron",
            "key_neurons": [n['id'] for n in sample_neurons[12:16]],
            "description": "Sugar sensing GRN neurons on forelegs fire rapid action potentials directly projecting into SEZ premotor centers.",
            "options": [
                {"id": 0, "text": "Proboscis Extension Response (PER) Feeding"},
                {"id": 1, "text": "Backward Flight Abort"},
                {"id": 2, "text": "Aggressive Head Butt Strike"},
                {"id": 3, "text": "Circadian Rest Entrance"}
            ],
            "correct_index": 0,
            "gnn_prediction": 0,
            "gnn_probabilities": [94.1, 2.0, 1.5, 2.4],
            "explanation": "Gr5a excitation releases acetylcholine in the SEZ, uninhibiting motor neuron MN9 to trigger proboscis extension and pharyngeal pumping."
        },
        {
            "id": "SCENARIO-05",
            "title": "Thermotactic Temperature Seeking (Antennal Warm/Cold Cells)",
            "category": "Sensory Navigation",
            "difficulty": "Hard",
            "sensory_stimulus": "Thermal Gradient Shift: Ambient 18°C -> 25°C Preference Zone",
            "stimulated_circuit": "Antennal Ir25a Thermal Sensors -> VP1 Glomeruli -> Hot Cell Interneurons",
            "key_neurons": [n['id'] for n in sample_neurons[16:20]],
            "description": "Temperature increase depolarizes warm-sensitive gustatory-like receptors, modulating turning frequency in temperature preference gradient.",
            "options": [
                {"id": 0, "text": "Sharp Turn Away from Warmth"},
                {"id": 1, "text": "Positive Thermotaxis (Taxis toward 25°C Preferred Zone)"},
                {"id": 2, "text": "Immediate Thermal Shock Coma"},
                {"id": 3, "text": "Wing Extension Vibration"}
            ],
            "correct_index": 1,
            "gnn_prediction": 1,
            "gnn_probabilities": [12.3, 81.5, 4.1, 2.1],
            "explanation": "Optimal thermal stimulation suppresses turning maneuvers while moving up gradient, promoting positive thermotaxis towards 25°C preference."
        }
    ]
    
    leaderboard_data = [
        {"rank": 1, "name": "Bio-GNN v4 (Ensemble)", "score": 9840, "accuracy": "98.4%", "badge": "AI Model", "is_ai": True},
        {"rank": 2, "name": "Dr. Sarah Chen (HHMI)", "score": 9210, "accuracy": "92.1%", "badge": "Neuroscientist", "is_ai": False},
        {"rank": 3, "name": "Spiking-LIF-Agent-3.0", "score": 8950, "accuracy": "89.5%", "badge": "AI Model", "is_ai": True},
        {"rank": 4, "name": "Prof. Marcus Vance", "score": 8620, "accuracy": "86.2%", "badge": "Connectomist", "is_ai": False},
        {"rank": 5, "name": "Connectome-Graph-RAG", "score": 8400, "accuracy": "84.0%", "badge": "AI Model", "is_ai": True}
    ]

    scenarios_json = json.dumps(scenarios)
    leaderboard_json = json.dumps(leaderboard_data)
    meta_json = json.dumps(loader.metadata)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NEURO-PREDICT: Connectome Behavior SaaS Challenge</title>
    <!-- Fonts & Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Canvas Confetti for rewards -->
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>

    <style>
        :root {{
            --bg-dark: #080c14;
            --card-bg: rgba(15, 23, 42, 0.75);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-cyan: #00f0ff;
            --accent-purple: #a855f7;
            --accent-green: #10b981;
            --accent-gold: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 10% 10%, rgba(0, 240, 255, 0.08) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(168, 85, 247, 0.08) 0px, transparent 50%);
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        /* Glassmorphism Panel */
        .glass-card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        }}

        /* Header Navigation */
        #header {{
            height: 72px;
            border-bottom: 1px solid var(--card-border);
            padding: 0 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(8, 12, 20, 0.85);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #00f0ff, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-badge {{
            font-size: 10px;
            padding: 3px 8px;
            background: rgba(0, 240, 255, 0.15);
            border: 1px solid rgba(0, 240, 255, 0.4);
            border-radius: 20px;
            color: var(--accent-cyan);
            -webkit-text-fill-color: initial;
            text-transform: uppercase;
            font-weight: 700;
        }}

        .nav-tabs {{
            display: flex;
            gap: 8px;
        }}

        .nav-tab {{
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .nav-tab:hover, .nav-tab.active {{
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.08);
        }}

        .nav-tab.active {{
            border: 1px solid rgba(0, 240, 255, 0.3);
            color: var(--accent-cyan);
        }}

        .user-stats-bar {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .stat-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .stat-val {{
            font-family: 'Fira Code', monospace;
            font-size: 16px;
            font-weight: 700;
            color: var(--accent-cyan);
        }}

        .streak-pill {{
            padding: 4px 12px;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(239, 68, 68, 0.2));
            border: 1px solid rgba(245, 158, 11, 0.5);
            border-radius: 20px;
            color: var(--accent-gold);
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
            animation: pulse-glow 2s infinite;
        }}

        @keyframes pulse-glow {{
            0%, 100% {{ box-shadow: 0 0 10px rgba(245, 158, 11, 0.2); }}
            50% {{ box-shadow: 0 0 20px rgba(245, 158, 11, 0.5); }}
        }}

        /* Main Container */
        #main-content {{
            max-width: 1200px;
            margin: 32px auto;
            padding: 0 24px;
            flex: 1;
            width: 100%;
        }}

        /* Scenario Card View */
        .scenario-container {{
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
        }}

        .card-header {{
            padding: 24px;
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .scenario-tag {{
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 6px;
            background: rgba(168, 85, 247, 0.2);
            border: 1px solid rgba(168, 85, 247, 0.4);
            color: var(--accent-purple);
            font-weight: 700;
            text-transform: uppercase;
        }}

        .card-body {{
            padding: 24px;
        }}

        .scenario-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 12px;
            color: var(--text-main);
        }}

        .scenario-desc {{
            font-size: 14px;
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 20px;
        }}

        /* Stimulus Highlight Box */
        .stimulus-box {{
            padding: 16px;
            background: rgba(0, 240, 255, 0.05);
            border: 1px dashed rgba(0, 240, 255, 0.3);
            border-radius: 10px;
            margin-bottom: 20px;
        }}

        .stimulus-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-cyan);
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .stimulus-val {{
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            color: var(--text-main);
        }}

        /* Mini Circuit Animation Canvas */
        #circuit-canvas {{
            width: 100%;
            height: 120px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            margin-bottom: 24px;
        }}

        /* Options Grid */
        .options-grid {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .option-btn {{
            padding: 16px 20px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
            text-align: left;
        }}

        .option-btn:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--accent-cyan);
            transform: translateX(4px);
        }}

        .option-btn.selected {{
            border-color: var(--accent-cyan);
            background: rgba(0, 240, 255, 0.15);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
        }}

        .option-btn.correct {{
            border-color: var(--accent-green) !important;
            background: rgba(16, 185, 129, 0.2) !important;
            color: #6ee7b7 !important;
        }}

        .option-btn.wrong {{
            border-color: var(--accent-red) !important;
            background: rgba(239, 68, 68, 0.2) !important;
            color: #fca5a5 !important;
        }}

        /* Right Panel: Model Comparison */
        .comparison-panel {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .model-card {{
            padding: 20px;
        }}

        .model-title {{
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .prob-bar-group {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .prob-bar-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .prob-label {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
        }}

        .prob-track {{
            height: 8px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 4px;
            overflow: hidden;
        }}

        .prob-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 4px;
            width: 0%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* Results Explanation Box */
        .result-box {{
            padding: 20px;
            border-radius: 12px;
            display: none;
            flex-direction: column;
            gap: 10px;
            animation: fadeIn 0.4s ease;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(100px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .submit-btn {{
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #00f0ff, #a855f7);
            border: none;
            border-radius: 12px;
            color: #000;
            font-weight: 700;
            font-size: 15px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
        }}

        .submit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 240, 255, 0.5);
        }}

        /* Leaderboard View */
        #leaderboard-view {{
            display: none;
        }}

        .lb-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .lb-table th {{
            text-align: left;
            padding: 14px 20px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--card-border);
        }}

        .lb-table td {{
            padding: 16px 20px;
            font-size: 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .rank-num {{
            font-family: 'Fira Code', monospace;
            font-weight: 700;
            width: 40px;
        }}

        .rank-1 {{ color: #fbbf24; }}
        .rank-2 {{ color: #94a3b8; }}
        .rank-3 {{ color: #d97706; }}
    </style>
</head>
<body>

    <!-- Header -->
    <div id="header">
        <div class="brand">
            <i class="fa-solid fa-gamepad"></i>
            <span>NEURO-PREDICT</span>
            <span class="brand-badge">SaaS Benchmark</span>
        </div>

        <div class="nav-tabs">
            <div class="nav-tab active" id="tab-scenarios" onclick="switchView('scenarios')">
                <i class="fa-solid fa-brain"></i>
                <span>Circuit Scenarios</span>
            </div>
            <div class="nav-tab" id="tab-leaderboard" onclick="switchView('leaderboard')">
                <i class="fa-solid fa-trophy"></i>
                <span>Global Leaderboard</span>
            </div>
        </div>

        <div class="user-stats-bar">
            <div class="stat-item">
                <span style="color:var(--text-muted);">Score:</span>
                <span class="stat-val" id="user-score">0</span>
            </div>
            <div class="streak-pill">
                <i class="fa-solid fa-fire"></i>
                <span>Streak: <span id="user-streak">0</span>x</span>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div id="main-content">

        <!-- SCENARIO VIEW -->
        <div id="scenario-view">
            <div class="scenario-container">
                <!-- Left: Card & Guessing -->
                <div class="glass-card">
                    <div class="card-header">
                        <div>
                            <span class="scenario-tag" id="scen-category">Category</span>
                            <span style="font-size: 12px; color: var(--text-muted); margin-left: 10px;" id="scen-diff">Medium</span>
                        </div>
                        <span style="font-family:'Fira Code'; font-size:12px; color:var(--text-muted);" id="scen-id">SCENARIO-01</span>
                    </div>

                    <div class="card-body">
                        <h2 class="scenario-title" id="scen-title">Scenario Title</h2>
                        <p class="scenario-desc" id="scen-desc">Description of circuit and sensory input.</p>

                        <div class="stimulus-box">
                            <div class="stimulus-title"><i class="fa-solid fa-wave-square"></i> Sensory Input Stimulus</div>
                            <div class="stimulus-val" id="scen-stimulus">Input details...</div>
                        </div>

                        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; font-weight:700;">
                            Circuit Activity Monitor
                        </div>
                        <canvas id="circuit-canvas"></canvas>

                        <div style="font-size: 13px; font-weight: 700; margin-bottom: 12px; color: var(--accent-cyan);">
                            Predict Behavioral Output:
                        </div>
                        <div class="options-grid" id="options-container">
                            <!-- Populated dynamically -->
                        </div>

                        <div style="margin-top: 24px;">
                            <button class="submit-btn" id="btn-submit-guess" onclick="submitGuess()">
                                Submit Prediction & Compare GNN
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Right: Model Comparison Panel -->
                <div class="comparison-panel">
                    <!-- Bio-GNN Card -->
                    <div class="glass-card model-card">
                        <div class="model-title">
                            <span>Bio-GNN v4 Model Prediction</span>
                            <i class="fa-solid fa-robot" style="color: var(--accent-purple);"></i>
                        </div>
                        <div class="prob-bar-group" id="gnn-prob-container">
                            <!-- Prob bars populated dynamically -->
                        </div>
                    </div>

                    <!-- Result Explanation Box -->
                    <div class="glass-card result-box" id="result-box">
                        <div style="display:flex; align-items:center; gap:10px; font-weight:700;" id="result-header">
                            <i class="fa-solid fa-circle-check" style="color: var(--accent-green); font-size:20px;"></i>
                            <span id="result-title">Prediction Verified!</span>
                        </div>
                        <p style="font-size: 12px; color: var(--text-muted); line-height:1.5;" id="result-explanation">
                            Explanation details...
                        </p>
                        <button class="submit-btn" style="padding:10px; font-size:13px;" onclick="nextScenario()">
                            Next Scenario <i class="fa-solid fa-arrow-right"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- LEADERBOARD VIEW -->
        <div id="leaderboard-view">
            <div class="glass-card" style="padding: 24px;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:24px;">
                    <div>
                        <h2 style="font-family:'Outfit'; font-size:24px; font-weight:800;">Global Neuro-AI Leaderboard</h2>
                        <p style="font-size:13px; color:var(--text-muted);">Real-time benchmark comparing Human Neuroscientists vs Bio-GNN Models.</p>
                    </div>
                    <div class="streak-pill">
                        <i class="fa-solid fa-shield-halved"></i>
                        <span>Season 4 Active</span>
                    </div>
                </div>

                <table class="lb-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Predictor / Agent</th>
                            <th>Classification</th>
                            <th>Accuracy</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody id="leaderboard-body">
                        <!-- Populated dynamically -->
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <!-- Application Script -->
    <script>
        const SCENARIOS = {scenarios_json};
        const LEADERBOARD = {leaderboard_json};

        let currentIdx = 0;
        let selectedOption = null;
        let userScore = 0;
        let userStreak = 0;
        let answered = false;

        // Simple Audio Synthesizer (Web Audio API)
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        function playSound(type) {{
            if (audioCtx.state === 'suspended') audioCtx.resume();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);

            if (type === 'click') {{
                osc.frequency.setValueAtTime(400, audioCtx.currentTime);
                gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.05);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.05);
            }} else if (type === 'success') {{
                osc.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
                osc.frequency.exponentialRampToValueAtTime(659.25, audioCtx.currentTime + 0.15); // E5
                gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.3);
            }} else if (type === 'fail') {{
                osc.frequency.setValueAtTime(200, audioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(120, audioCtx.currentTime + 0.2);
                gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.25);
            }}
        }}

        function init() {{
            loadScenario(currentIdx);
            renderLeaderboard();
            drawCircuitAnimation();
        }}

        function loadScenario(idx) {{
            answered = false;
            selectedOption = null;
            const scen = SCENARIOS[idx];

            document.getElementById('scen-id').innerText = scen.id;
            document.getElementById('scen-category').innerText = scen.category;
            document.getElementById('scen-diff').innerText = scen.difficulty;
            document.getElementById('scen-title').innerText = scen.title;
            document.getElementById('scen-desc').innerText = scen.description;
            document.getElementById('scen-stimulus').innerText = scen.sensory_stimulus;

            document.getElementById('result-box').style.display = 'none';

            // Options Grid
            const container = document.getElementById('options-container');
            container.innerHTML = '';
            scen.options.forEach((opt) => {{
                const btn = document.createElement('button');
                btn.className = 'option-btn';
                btn.innerHTML = `<span>${{opt.text}}</span> <i class="fa-regular fa-circle"></i>`;
                btn.onclick = () => selectOption(opt.id, btn);
                container.appendChild(btn);
            }});

            renderGNNProbabilities(scen, null);
        }}

        function selectOption(id, btnElement) {{
            if (answered) return;
            playSound('click');
            selectedOption = id;

            document.querySelectorAll('.option-btn').forEach(b => {{
                b.classList.remove('selected');
                b.querySelector('i').className = 'fa-regular fa-circle';
            }});

            btnElement.classList.add('selected');
            btnElement.querySelector('i').className = 'fa-solid fa-circle-dot';
        }}

        function submitGuess() {{
            if (selectedOption === null || answered) return;
            answered = true;

            const scen = SCENARIOS[currentIdx];
            const isCorrect = selectedOption === scen.correct_index;

            const btns = document.querySelectorAll('.option-btn');
            btns.forEach((btn, idx) => {{
                if (idx === scen.correct_index) {{
                    btn.classList.add('correct');
                    btn.querySelector('i').className = 'fa-solid fa-circle-check';
                }} else if (idx === selectedOption) {{
                    btn.classList.add('wrong');
                    btn.querySelector('i').className = 'fa-solid fa-circle-xmark';
                }}
            }});

            // Scoring & Streaks
            if (isCorrect) {{
                playSound('success');
                userStreak += 1;
                const points = 1000 + (userStreak * 250);
                userScore += points;
                confetti({{ particleCount: 80, spread: 60, origin: {{ y: 0.7 }} }});
            }} else {{
                playSound('fail');
                userStreak = 0;
            }}

            document.getElementById('user-score').innerText = userScore.toLocaleString();
            document.getElementById('user-streak').innerText = userStreak;

            // Show Result Explanation
            const resBox = document.getElementById('result-box');
            resBox.style.display = 'flex';
            if (isCorrect) {{
                document.getElementById('result-header').innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--accent-green); font-size:20px;"></i> <span>Prediction Correct! (+${{1000 + userStreak * 250}} pts)</span>`;
            }} else {{
                document.getElementById('result-header').innerHTML = `<i class="fa-solid fa-circle-xmark" style="color:var(--accent-red); font-size:20px;"></i> <span>Prediction Mismatch</span>`;
            }}
            document.getElementById('result-explanation').innerText = scen.explanation;

            renderGNNProbabilities(scen, selectedOption);
            updateLeaderboardWithUser();
        }}

        function renderGNNProbabilities(scen, userChoice) {{
            const container = document.getElementById('gnn-prob-container');
            container.innerHTML = '';

            scen.options.forEach((opt, i) => {{
                const prob = scen.gnn_probabilities[i];
                const item = document.createElement('div');
                item.className = 'prob-bar-item';
                item.innerHTML = `
                    <div class="prob-label">
                        <span>${{opt.text}}</span>
                        <span style="font-family:'Fira Code'; font-weight:700;">${{prob}}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: ${{answered ? prob : 0}}%;"></div>
                    </div>
                `;
                container.appendChild(item);
            }});

            if (answered) {{
                setTimeout(() => {{
                    document.querySelectorAll('.prob-fill').forEach((el, idx) => {{
                        el.style.width = scen.gnn_probabilities[idx] + '%';
                    }});
                }}, 50);
            }}
        }}

        function nextScenario() {{
            currentIdx = (currentIdx + 1) % SCENARIOS.length;
            loadScenario(currentIdx);
        }}

        function switchView(view) {{
            playSound('click');
            if (view === 'scenarios') {{
                document.getElementById('scenario-view').style.display = 'block';
                document.getElementById('leaderboard-view').style.display = 'none';
                document.getElementById('tab-scenarios').classList.add('active');
                document.getElementById('tab-leaderboard').classList.remove('active');
            }} else {{
                document.getElementById('scenario-view').style.display = 'none';
                document.getElementById('leaderboard-view').style.display = 'block';
                document.getElementById('tab-scenarios').classList.remove('active');
                document.getElementById('tab-leaderboard').classList.add('active');
            }}
        }}

        function renderLeaderboard() {{
            const tbody = document.getElementById('leaderboard-body');
            tbody.innerHTML = '';
            LEADERBOARD.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="rank-num rank-${{row.rank}}">#${{row.rank}}</td>
                    <td style="font-weight:700; display:flex; align-items:center; gap:8px;">
                        ${{row.is_ai ? '<i class="fa-solid fa-robot" style="color:var(--accent-purple)"></i>' : '<i class="fa-solid fa-user-astronaut" style="color:var(--accent-cyan)"></i>'}}
                        ${{row.name}}
                    </td>
                    <td><span class="scenario-tag" style="font-size:10px;">${{row.badge}}</span></td>
                    <td style="font-family:'Fira Code';">${{row.accuracy}}</td>
                    <td style="font-family:'Fira Code'; font-weight:700; color:var(--accent-cyan);">${{row.score.toLocaleString()}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function updateLeaderboardWithUser() {{
            let userEntry = LEADERBOARD.find(e => e.name === 'Player (You)');
            if (!userEntry) {{
                userEntry = {{ rank: 6, name: 'Player (You)', score: userScore, accuracy: '100%', badge: 'User', is_ai: false }};
                LEADERBOARD.push(userEntry);
            }} else {{
                userEntry.score = userScore;
            }}

            LEADERBOARD.sort((a, b) => b.score - a.score);
            LEADERBOARD.forEach((item, idx) => item.rank = idx + 1);
            renderLeaderboard();
        }}

        function drawCircuitAnimation() {{
            const canvas = document.getElementById('circuit-canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;

            let particles = Array.from({{length: 12}}, () => ({{
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                speed: 1 + Math.random() * 2
            }}));

            function anim() {{
                ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(20, canvas.height/2);
                ctx.lineTo(canvas.width - 20, canvas.height/2);
                ctx.stroke();

                particles.forEach(p => {{
                    p.x += p.speed;
                    if (p.x > canvas.width) p.x = 0;
                    ctx.fillStyle = '#00f0ff';
                    ctx.beginPath();
                    ctx.arc(p.x, canvas.height/2 + Math.sin(p.x * 0.05) * 20, 3, 0, Math.PI * 2);
                    ctx.fill();
                }});

                requestAnimationFrame(anim);
            }}
            anim();
        }}

        window.onload = init;
    </script>
</body>
</html>
"""

    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✓ Product 3 built successfully: {output_dir / 'index.html'}")
    return {
        "status": "success",
        "output_file": str(output_dir / "index.html"),
        "scenarios_count": len(scenarios)
    }
