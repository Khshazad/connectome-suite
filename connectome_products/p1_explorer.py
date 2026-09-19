"""
Product 1: Interactive 3D WebGL Connectome Explorer
===================================================
Generates a standalone, feature-complete WebGL Three.js interactive 3D brain explorer.
Features:
- 3D Interactive Point Cloud of Neurons colored by cell type
- Synaptic Edge Wireframes with excitatory/inhibitory coloring
- Interactive Controls: Node filtering, Search by ID/Type, Camera controls
- Real-time Pulse Signal Propagation Animation (particle flow along edges)
- Click-to-Lesion (disabling nodes, calculating downstream path disruption, warning highlights)
- Glassmorphism dark mode UI styling & dashboard layout
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def build_product_1(loader, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare data from loader
    neurons_df = loader.neurons.copy()
    synapses_df = loader.synapses.copy()
    
    # Pre-calculate degree for neurons
    in_degrees = synapses_df['target'].value_counts().to_dict()
    out_degrees = synapses_df['source'].value_counts().to_dict()
    
    neurons_list = []
    for _, row in neurons_df.iterrows():
        nid = str(row['id'])
        neurons_list.append({
            "id": nid,
            "name": str(row['name']),
            "type": str(row['type']),
            "region": str(row['region']),
            "nt": str(row['neurotransmitter']),
            "x": round(float(row['x']), 2),
            "y": round(float(row['y']), 2),
            "z": round(float(row['z']), 2),
            "gene": round(float(row['gene_expression']), 3),
            "in_deg": in_degrees.get(nid, 0),
            "out_deg": out_degrees.get(nid, 0)
        })
        
    synapses_list = []
    for _, row in synapses_df.iterrows():
        synapses_list.append({
            "s": str(row['source']),
            "t": str(row['target']),
            "w": round(float(row['weight']), 2),
            "nt": str(row['neurotransmitter'])
        })
        
    neurons_json = json.dumps(neurons_list)
    synapses_json = json.dumps(synapses_list)
    meta_json = json.dumps(loader.metadata)
    
    # 2. Build HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fruit Fly Connectome - Interactive 3D WebGL Explorer</title>
    <!-- Fonts & Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Three.js and OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <style>
        :root {{
            --bg-dark: #070a12;
            --panel-bg: rgba(13, 18, 30, 0.78);
            --panel-border: rgba(255, 255, 255, 0.12);
            --accent-cyan: #00f0ff;
            --accent-purple: #a855f7;
            --accent-pink: #ff2a6d;
            --accent-gold: #ffb703;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}

        body, html {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: var(--bg-dark);
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
        }}

        #canvas-container {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}

        /* Glassmorphism UI Panels */
        .glass-panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        }}

        /* Top Header Navigation */
        #top-header {{
            position: absolute;
            top: 16px;
            left: 16px;
            right: 16px;
            height: 64px;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
        }}

        .brand-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #00f0ff, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-title i {{
            -webkit-text-fill-color: initial;
            color: var(--accent-cyan);
            font-size: 22px;
        }}

        .header-stats {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}

        .stat-badge {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }}

        .stat-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
        }}

        .stat-value {{
            font-family: 'Fira Code', monospace;
            font-size: 15px;
            font-weight: 600;
            color: var(--accent-cyan);
        }}

        /* Floating Sidebar Left */
        #sidebar-left {{
            position: absolute;
            top: 96px;
            left: 16px;
            width: 340px;
            bottom: 24px;
            z-index: 10;
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 20px;
            overflow-y: auto;
        }}

        #sidebar-left::-webkit-scrollbar {{
            width: 4px;
        }}
        #sidebar-left::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }}

        .panel-section-title {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        /* Search Input */
        .search-box {{
            position: relative;
            margin-bottom: 12px;
        }}

        .search-box input {{
            width: 100%;
            padding: 10px 14px 10px 38px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 13px;
            outline: none;
            transition: all 0.2s ease;
        }}

        .search-box input:focus {{
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.25);
        }}

        .search-box i {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 14px;
        }}

        /* Cell Type Filters */
        .type-filter-list {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .filter-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 12px;
        }}

        .filter-item:hover {{
            background: rgba(255, 255, 255, 0.08);
        }}

        .filter-item.active {{
            border-color: rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.1);
        }}

        .filter-color-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            display: inline-block;
        }}

        /* Control Buttons */
        .btn {{
            width: 100%;
            padding: 10px 16px;
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(168, 85, 247, 0.2));
            border: 1px solid rgba(0, 240, 255, 0.4);
            border-radius: 8px;
            color: var(--text-main);
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s ease;
        }}

        .btn:hover {{
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.4), rgba(168, 85, 247, 0.4));
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
            transform: translateY(-1px);
        }}

        .btn-danger {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(255, 42, 109, 0.2));
            border-color: rgba(239, 68, 68, 0.5);
        }}

        .btn-danger:hover {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.4), rgba(255, 42, 109, 0.4));
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
        }}

        /* Floating Inspector Card Right */
        #inspector-panel {{
            position: absolute;
            top: 96px;
            right: 16px;
            width: 340px;
            z-index: 10;
            padding: 20px;
            display: none;
        }}

        .inspector-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--panel-border);
        }}

        .inspector-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 700;
            color: var(--accent-cyan);
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .info-label {{
            color: var(--text-muted);
        }}

        .info-val {{
            font-family: 'Fira Code', monospace;
            font-weight: 500;
        }}

        /* Lesion Warning Banner */
        #lesion-banner {{
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            padding: 14px 28px;
            background: rgba(239, 68, 68, 0.25);
            border: 1px solid rgba(239, 68, 68, 0.6);
            backdrop-filter: blur(12px);
            border-radius: 30px;
            display: none;
            align-items: center;
            gap: 16px;
            color: #fca5a5;
            font-size: 13px;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.4);
        }}

        /* Tooltip */
        #tooltip {{
            position: absolute;
            z-index: 20;
            padding: 8px 12px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid var(--accent-cyan);
            border-radius: 6px;
            color: white;
            font-size: 11px;
            pointer-events: none;
            display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}

        /* Quick Controls overlay bottom left */
        #viewport-controls {{
            position: absolute;
            bottom: 24px;
            left: 380px;
            z-index: 10;
            display: flex;
            gap: 8px;
        }}

        .icon-btn {{
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .icon-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}

        /* Degree Canvas */
        #degree-chart {{
            width: 100%;
            height: 80px;
            margin-top: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
        }}
    </style>
</head>
<body>

    <!-- 3D Canvas Container -->
    <div id="canvas-container"></div>

    <!-- Tooltip -->
    <div id="tooltip"></div>

    <!-- Top Navigation -->
    <div id="top-header" class="glass-panel">
        <div class="brand-title">
            <i class="fa-solid fa-brain"></i>
            <span>CONNECTOME 3D EXPLORER</span>
        </div>
        <div class="header-stats">
            <div class="stat-badge">
                <span class="stat-label">Neurons</span>
                <span class="stat-value" id="stat-neuron-count">0</span>
            </div>
            <div class="stat-badge">
                <span class="stat-label">Synapses</span>
                <span class="stat-value" id="stat-synapse-count">0</span>
            </div>
            <div class="stat-badge">
                <span class="stat-label">Active Signals</span>
                <span class="stat-value" id="stat-active-signals" style="color: var(--accent-gold);">0</span>
            </div>
            <div class="stat-badge">
                <span class="stat-label">Lesioned</span>
                <span class="stat-value" id="stat-lesioned-count" style="color: var(--accent-red);">0</span>
            </div>
        </div>
    </div>

    <!-- Left Control Panel -->
    <div id="sidebar-left" class="glass-panel">
        <div>
            <div class="panel-section-title">
                <span>Search & Filter</span>
                <i class="fa-solid fa-sliders"></i>
            </div>
            <div class="search-box">
                <i class="fa-solid fa-magnifying-glass"></i>
                <input type="text" id="search-input" placeholder="Search Neuron ID or Type...">
            </div>

            <div class="panel-section-title" style="margin-top: 16px;">
                <span>Cell Types</span>
            </div>
            <div class="type-filter-list" id="type-filter-container">
                <!-- Populated dynamically -->
            </div>
        </div>

        <div style="margin-top: 10px;">
            <div class="panel-section-title">
                <span>Signal Propagation</span>
                <i class="fa-solid fa-bolt"></i>
            </div>
            <button class="btn" id="btn-fire-pulse">
                <i class="fa-solid fa-bolt-lightning"></i>
                <span>Fire Stimulus Pulse</span>
            </button>
            <div style="margin-top: 8px; font-size: 11px; color: var(--text-muted); text-align: center;">
                Select a neuron or click button to pulse random sensory hub
            </div>
        </div>

        <div style="margin-top: 10px;">
            <div class="panel-section-title">
                <span>Degree Distribution</span>
            </div>
            <canvas id="degree-chart"></canvas>
        </div>

        <div style="margin-top: auto; display: flex; flex-direction: column; gap: 8px;">
            <button class="btn btn-danger" id="btn-reset-lesions" style="display: none;">
                <i class="fa-solid fa-rotate-left"></i>
                <span>Restore All Lesioned Nodes</span>
            </button>
        </div>
    </div>

    <!-- Right Inspector Panel -->
    <div id="inspector-panel" class="glass-panel">
        <div class="inspector-header">
            <span class="inspector-title" id="insp-id">Neuron Inspector</span>
            <i class="fa-solid fa-xmark" id="btn-close-inspector" style="cursor: pointer; color: var(--text-muted);"></i>
        </div>
        <div id="inspector-details">
            <div class="info-row"><span class="info-label">Name</span><span class="info-val" id="insp-name">-</span></div>
            <div class="info-row"><span class="info-label">Cell Type</span><span class="info-val" id="insp-type" style="color: var(--accent-cyan);">-</span></div>
            <div class="info-row"><span class="info-label">Brain Region</span><span class="info-val" id="insp-region">-</span></div>
            <div class="info-row"><span class="info-label">Neurotransmitter</span><span class="info-val" id="insp-nt">-</span></div>
            <div class="info-row"><span class="info-label">In-Degree</span><span class="info-val" id="insp-indeg">-</span></div>
            <div class="info-row"><span class="info-label">Out-Degree</span><span class="info-val" id="insp-outdeg">-</span></div>
            <div class="info-row"><span class="info-label">Gene Exp.</span><span class="info-val" id="insp-gene">-</span></div>
            <div class="info-row"><span class="info-label">Coordinates</span><span class="info-val" id="insp-pos">-</span></div>
        </div>

        <div style="margin-top: 16px; display: flex; flex-direction: column; gap: 10px;">
            <button class="btn" id="btn-insp-pulse">
                <i class="fa-solid fa-bolt"></i>
                <span>Propagate Pulse from Here</span>
            </button>
            <button class="btn btn-danger" id="btn-insp-lesion">
                <i class="fa-solid fa-skull-crossbones"></i>
                <span>Lesion & Disrupt Pathway</span>
            </button>
        </div>
    </div>

    <!-- Viewport Quick Controls -->
    <div id="viewport-controls">
        <div class="icon-btn" id="btn-reset-cam" title="Reset Camera View"><i class="fa-solid fa-arrows-to-eye"></i></div>
        <div class="icon-btn" id="btn-toggle-wireframe" title="Toggle Brain Envelope Wireframe"><i class="fa-solid fa-globe"></i></div>
        <div class="icon-btn" id="btn-toggle-edges" title="Toggle Synapse Edges"><i class="fa-solid fa-diagram-project"></i></div>
        <div class="icon-btn" id="btn-toggle-rotation" title="Toggle Auto-Rotation"><i class="fa-solid fa-rotate"></i></div>
    </div>

    <!-- Lesion Warning Banner -->
    <div id="lesion-banner">
        <i class="fa-solid fa-triangle-exclamation" style="font-size: 20px;"></i>
        <div>
            <strong id="lesion-banner-title">Pathway Disruption Active</strong>
            <div id="lesion-banner-desc" style="font-size: 11px;">1 node lesioned. 14 downstream paths disrupted.</div>
        </div>
    </div>

    <!-- Embedded Data & Application Script -->
    <script>
        const NEURON_DATA = {neurons_json};
        const SYNAPSE_DATA = {synapses_json};
        const METADATA = {meta_json};

        // Palette mapping for cell types
        const TYPE_COLORS = {{
            'sensory': '#00f0ff',
            'interneuron': '#a855f7',
            'motor': '#ff2a6d',
            'central_complex': '#ffb703',
            'kenyon_cell': '#10b981',
            'giant_fiber': '#00ff9f'
        }};

        const DEFAULT_COLOR = '#94a3b8';

        // State variables
        let scene, camera, renderer, controls;
        let neuronMeshes = [];
        let neuronMap = new Map();
        let synapseLines = null;
        let brainWireframe = null;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2();
        
        let selectedNeuron = null;
        let lesionedIds = new Set();
        let activePulses = [];
        let autoRotate = false;
        let activeFilters = new Set(Object.keys(TYPE_COLORS));
        let showEdges = true;

        // Initialize App
        function init() {{
            document.getElementById('stat-neuron-count').innerText = NEURON_DATA.length.toLocaleString();
            document.getElementById('stat-synapse-count').innerText = SYNAPSE_DATA.length.toLocaleString();

            NEURON_DATA.forEach(n => neuronMap.set(n.id, n));

            setupThreeJS();
            createBrainEnvelope();
            createNeuronNodes();
            createSynapseEdges();
            setupFilterUI();
            drawDegreeChart();
            setupEventListeners();
            animate();
        }}

        function setupThreeJS() {{
            const container = document.getElementById('canvas-container');
            scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x070a12, 0.002);

            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 50, 220);

            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setClearColor(0x070a12, 1);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxDistance = 500;
            controls.minDistance = 20;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const dirLight1 = new THREE.DirectionalLight(0x00f0ff, 0.8);
            dirLight1.position.set(100, 100, 100);
            scene.add(dirLight1);

            const dirLight2 = new THREE.DirectionalLight(0xa855f7, 0.8);
            dirLight2.position.set(-100, -100, -100);
            scene.add(dirLight2);
        }}

        function createBrainEnvelope() {{
            const geometry = new THREE.IcosahedronGeometry(130, 2);
            const material = new THREE.MeshBasicMaterial({{
                color: 0x00f0ff,
                wireframe: true,
                transparent: true,
                opacity: 0.04
            }});
            brainWireframe = new THREE.Mesh(geometry, material);
            scene.add(brainWireframe);
        }}

        function createNeuronNodes() {{
            const sphereGeo = new THREE.SphereGeometry(1.6, 12, 12);
            
            NEURON_DATA.forEach(n => {{
                const colorHex = TYPE_COLORS[n.type] || DEFAULT_COLOR;
                const mat = new THREE.MeshPhongMaterial({{
                    color: new THREE.Color(colorHex),
                    emissive: new THREE.Color(colorHex),
                    emissiveIntensity: 0.4,
                    shininess: 80,
                    transparent: true,
                    opacity: 0.95
                }});

                const mesh = new THREE.Mesh(sphereGeo, mat);
                mesh.position.set(n.x, n.y, n.z);
                mesh.userData = n;
                scene.add(mesh);
                neuronMeshes.push(mesh);
            }});
        }}

        function createSynapseEdges() {{
            const positions = [];
            const colors = [];

            SYNAPSE_DATA.forEach(s => {{
                const src = neuronMap.get(s.s);
                const tgt = neuronMap.get(s.t);
                if (src && tgt) {{
                    positions.push(src.x, src.y, src.z);
                    positions.push(tgt.x, tgt.y, tgt.z);

                    const color = s.nt === 'excitatory' ? new THREE.Color(0x00f0ff) : new THREE.Color(0xff2a6d);
                    colors.push(color.r, color.g, color.b);
                    colors.push(color.r, color.g, color.b);
                }}
            }});

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const material = new THREE.LineBasicMaterial({{
                vertexColors: true,
                transparent: true,
                opacity: 0.12,
                blending: THREE.AdditiveBlending
            }});

            synapseLines = new THREE.LineSegments(geometry, material);
            scene.add(synapseLines);
        }}

        function setupFilterUI() {{
            const container = document.getElementById('type-filter-container');
            container.innerHTML = '';

            const types = Object.keys(TYPE_COLORS);
            types.forEach(t => {{
                const color = TYPE_COLORS[t];
                const count = NEURON_DATA.filter(n => n.type === t).length;

                const item = document.createElement('div');
                item.className = 'filter-item active';
                item.dataset.type = t;
                item.innerHTML = `
                    <div style="display:flex; align-items:center;">
                        <span class="filter-color-dot" style="background:${{color}};"></span>
                        <span style="text-transform: capitalize;">${{t.replace('_', ' ')}}</span>
                    </div>
                    <span style="font-family:'Fira Code'; color:var(--text-muted); font-size:11px;">${{count}}</span>
                `;

                item.addEventListener('click', () => {{
                    if (activeFilters.has(t)) {{
                        activeFilters.delete(t);
                        item.classList.remove('active');
                    }} else {{
                        activeFilters.add(t);
                        item.classList.add('active');
                    }}
                    applyFilters();
                }});

                container.appendChild(item);
            }});
        }}

        function applyFilters() {{
            const query = document.getElementById('search-input').value.toLowerCase().trim();

            neuronMeshes.forEach(mesh => {{
                const n = mesh.userData;
                const matchesType = activeFilters.has(n.type);
                const matchesQuery = !query || n.id.toLowerCase().includes(query) || n.type.toLowerCase().includes(query) || n.region.toLowerCase().includes(query);

                const visible = matchesType && matchesQuery;
                mesh.visible = visible;
            }});
        }}

        function drawDegreeChart() {{
            const canvas = document.getElementById('degree-chart');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;

            const degrees = NEURON_DATA.map(n => n.in_deg + n.out_deg);
            const maxDeg = Math.max(...degrees, 1);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#00f0ff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();

            const step = canvas.width / (degrees.length - 1);
            degrees.forEach((d, i) => {{
                const x = i * step;
                const y = canvas.height - (d / maxDeg) * (canvas.height - 10) - 5;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();

            // Gradient fill under line
            ctx.lineTo(canvas.width, canvas.height);
            ctx.lineTo(0, canvas.height);
            const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
            grad.addColorStop(0, 'rgba(0, 240, 255, 0.3)');
            grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
            ctx.fillStyle = grad;
            ctx.fill();
        }}

        function firePulse(startNodeId) {{
            let rootId = startNodeId;
            if (!rootId) {{
                const sensoryNodes = NEURON_DATA.filter(n => n.type === 'sensory' && !lesionedIds.has(n.id));
                if (sensoryNodes.length > 0) {{
                    rootId = sensoryNodes[Math.floor(Math.random() * sensoryNodes.length)].id;
                }} else {{
                    rootId = NEURON_DATA[0].id;
                }}
            }}

            const root = neuronMap.get(rootId);
            if (!root || lesionedIds.has(rootId)) return;

            // Find connected targets
            const outgoingSynapses = SYNAPSE_DATA.filter(s => s.s === rootId && !lesionedIds.has(s.t));
            
            outgoingSynapses.forEach(syn => {{
                const target = neuronMap.get(syn.t);
                if (target) {{
                    // Create glowing pulse particle
                    const pGeo = new THREE.SphereGeometry(1.2, 8, 8);
                    const pMat = new THREE.MeshBasicMaterial({{
                        color: 0xffb703,
                        transparent: true,
                        opacity: 1.0
                    }});
                    const particle = new THREE.Mesh(pGeo, pMat);
                    particle.position.set(root.x, root.y, root.z);
                    scene.add(particle);

                    activePulses.push({{
                        mesh: particle,
                        start: new THREE.Vector3(root.x, root.y, root.z),
                        end: new THREE.Vector3(target.x, target.y, target.z),
                        progress: 0,
                        speed: 0.025 + Math.random() * 0.015,
                        targetId: target.id
                    }});
                }}
            }});

            document.getElementById('stat-active-signals').innerText = activePulses.length;
        }}

        function lesionNeuron(neuronId) {{
            lesionedIds.add(neuronId);
            const targetMesh = neuronMeshes.find(m => m.userData.id === neuronId);
            if (targetMesh) {{
                targetMesh.material.color.setHex(0xef4444);
                targetMesh.material.emissive.setHex(0xef4444);
                targetMesh.material.opacity = 0.3;
                targetMesh.scale.set(0.7, 0.7, 0.7);
            }}

            // Calculate downstream affected nodes
            const affectedDownstream = new Set();
            function traceDownstream(nid) {{
                SYNAPSE_DATA.filter(s => s.s === nid).forEach(syn => {{
                    if (!affectedDownstream.has(syn.t) && !lesionedIds.has(syn.t)) {{
                        affectedDownstream.add(syn.t);
                        traceDownstream(syn.t);
                    }}
                }});
            }}
            traceDownstream(neuronId);

            // Highlight affected downstream nodes
            affectedDownstream.forEach(downId => {{
                const mesh = neuronMeshes.find(m => m.userData.id === downId);
                if (mesh) {{
                    mesh.material.color.setHex(0xffb703);
                    mesh.material.emissive.setHex(0xffb703);
                }}
            }});

            // Update banner & UI
            document.getElementById('stat-lesioned-count').innerText = lesionedIds.size;
            document.getElementById('btn-reset-lesions').style.display = 'flex';

            const banner = document.getElementById('lesion-banner');
            banner.style.display = 'flex';
            document.getElementById('lesion-banner-desc').innerText = 
                `${{lesionedIds.size}} node(s) disabled. ${{affectedDownstream.size}} downstream neuron pathway(s) disrupted.`;
        }}

        function resetLesions() {{
            lesionedIds.clear();
            neuronMeshes.forEach(mesh => {{
                const n = mesh.userData;
                const colorHex = TYPE_COLORS[n.type] || DEFAULT_COLOR;
                mesh.material.color.set(colorHex);
                mesh.material.emissive.set(colorHex);
                mesh.material.opacity = 0.95;
                mesh.scale.set(1, 1, 1);
            }});

            document.getElementById('stat-lesioned-count').innerText = '0';
            document.getElementById('btn-reset-lesions').style.display = 'none';
            document.getElementById('lesion-banner').style.display = 'none';
        }}

        function selectNeuron(n) {{
            selectedNeuron = n;
            document.getElementById('insp-id').innerText = n.id;
            document.getElementById('insp-name').innerText = n.name;
            document.getElementById('insp-type').innerText = n.type.toUpperCase();
            document.getElementById('insp-region').innerText = n.region.replace('_', ' ');
            document.getElementById('insp-nt').innerText = n.nt;
            document.getElementById('insp-indeg').innerText = n.in_deg;
            document.getElementById('insp-outdeg').innerText = n.out_deg;
            document.getElementById('insp-gene').innerText = n.gene;
            document.getElementById('insp-pos').innerText = `${{n.x}}, ${{n.y}}, ${{n.z}}`;

            document.getElementById('inspector-panel').style.display = 'block';

            // Focus camera smooth zoom
            const targetPos = new THREE.Vector3(n.x, n.y, n.z);
            controls.target.copy(targetPos);
        }}

        function setupEventListeners() {{
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            document.getElementById('search-input').addEventListener('input', applyFilters);

            document.getElementById('btn-fire-pulse').addEventListener('click', () => firePulse(null));

            document.getElementById('btn-insp-pulse').addEventListener('click', () => {{
                if (selectedNeuron) firePulse(selectedNeuron.id);
            }});

            document.getElementById('btn-insp-lesion').addEventListener('click', () => {{
                if (selectedNeuron) lesionNeuron(selectedNeuron.id);
            }});

            document.getElementById('btn-reset-lesions').addEventListener('click', resetLesions);
            document.getElementById('btn-close-inspector').addEventListener('click', () => {{
                document.getElementById('inspector-panel').style.display = 'none';
            }});

            // Viewport buttons
            document.getElementById('btn-reset-cam').addEventListener('click', () => {{
                camera.position.set(0, 50, 220);
                controls.target.set(0, 0, 0);
            }});

            document.getElementById('btn-toggle-wireframe').addEventListener('click', () => {{
                brainWireframe.visible = !brainWireframe.visible;
            }});

            document.getElementById('btn-toggle-edges').addEventListener('click', () => {{
                showEdges = !showEdges;
                synapseLines.visible = showEdges;
            }});

            document.getElementById('btn-toggle-rotation').addEventListener('click', () => {{
                autoRotate = !autoRotate;
            }});

            // Raycasting Click & Hover
            window.addEventListener('pointermove', (event) => {{
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(neuronMeshes);

                const tooltip = document.getElementById('tooltip');
                if (intersects.length > 0 && intersects[0].object.visible) {{
                    const n = intersects[0].object.userData;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 15) + 'px';
                    tooltip.style.top = (event.clientY + 15) + 'px';
                    tooltip.innerHTML = `<strong>${{n.id}}</strong> (${{n.type}})<br>Region: ${{n.region}}<br>Neurotransmitter: ${{n.nt}}`;
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }});

            window.addEventListener('click', (event) => {{
                // Ignore clicks on UI elements
                if (event.target.closest('.glass-panel') || event.target.closest('#viewport-controls')) return;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(neuronMeshes);

                if (intersects.length > 0 && intersects[0].object.visible) {{
                    selectNeuron(intersects[0].object.userData);
                }}
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();

            if (autoRotate) {{
                scene.rotation.y += 0.003;
            }}

            // Update active pulses
            for (let i = activePulses.length - 1; i >= 0; i--) {{
                const p = activePulses[i];
                p.progress += p.speed;
                p.mesh.position.lerpVectors(p.start, p.end, p.progress);

                if (p.progress >= 1.0) {{
                    scene.remove(p.mesh);
                    activePulses.splice(i, 1);
                    
                    // Cascade pulse to next layer with probability
                    if (Math.random() < 0.4) {{
                        firePulse(p.targetId);
                    }}
                }}
            }}
            document.getElementById('stat-active-signals').innerText = activePulses.length;

            renderer.render(scene, camera);
        }}

        window.onload = init;
    </script>
</body>
</html>
"""

    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✓ Product 1 built successfully: {output_dir / 'index.html'}")
    return {
        "status": "success",
        "output_file": str(output_dir / "index.html"),
        "neurons_count": len(neurons_list),
        "synapses_count": len(synapses_list)
    }
