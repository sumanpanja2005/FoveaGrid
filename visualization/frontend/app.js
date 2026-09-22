// FoveaGrid Unified 2.5D LiDAR Roadside Elevation Map Viewer
// Implements real-time WebSocket stream ingestion, dynamic 2.5D DEM plane deformation,
// false-color elevation vertex gradient, draped curbs, pothole wells, and 3D obstacle bounding boxes.

let scene, camera, renderer, controls;
let elevationMesh, elevationPoints, elevationGeometry, elevationMaterial;
let curbsGroup, potholesGroup, obstaclesGroup, gridHelper;
let ws = null;
let currentStyle = 'surface';
let isPaused = false;
let lastFrameTime = performance.now();
let fpsCounter = 0;
let liveFps = 15;

// Color Gradient Definitions (Cool to Warm: Pothole -> Road -> Pedestrian -> Vehicle -> Pole)
const COLOR_STOPS = [
    { zNorm: 0.00, color: new THREE.Color(0x081d58) }, // Deep Navy (pothole depression)
    { zNorm: 0.15, color: new THREE.Color(0x1d91c0) }, // Electric Cyan (ground road base)
    { zNorm: 0.30, color: new THREE.Color(0x41b6c4) }, // Turquoise (sidewalk / road)
    { zNorm: 0.45, color: new THREE.Color(0x1dd1a1) }, // Seafoam Green (pavement)
    { zNorm: 0.60, color: new THREE.Color(0xfeca57) }, // Gold (curb / low bumper)
    { zNorm: 0.75, color: new THREE.Color(0xff9f43) }, // Amber (pedestrian / hood)
    { zNorm: 0.90, color: new THREE.Color(0xff6b6b) }, // Orange (vehicle roof)
    { zNorm: 1.00, color: new THREE.Color(0xbd0026) }  // Crimson (tall poles, infrastructure)
];

function getElevationColor(z, minZ, maxZ) {
    const range = Math.max(0.5, maxZ - minZ);
    const t = Math.max(0.0, Math.min(1.0, (z - minZ) / range));

    for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
        if (t >= COLOR_STOPS[i].zNorm && t <= COLOR_STOPS[i + 1].zNorm) {
            const span = COLOR_STOPS[i + 1].zNorm - COLOR_STOPS[i].zNorm;
            const factor = (t - COLOR_STOPS[i].zNorm) / span;
            const c = new THREE.Color();
            c.lerpColors(COLOR_STOPS[i].color, COLOR_STOPS[i + 1].color, factor);
            return c;
        }
    }
    return COLOR_STOPS[COLOR_STOPS.length - 1].color.clone();
}

function initThree() {
    const container = document.getElementById('three-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Scene & Background
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05080f);
    scene.fog = new THREE.FogExp2(0x05080f, 0.008);

    // 2. Camera (Isometric Perspective Default)
    camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    setCameraPreset('isometric');

    // 3. WebGL Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(renderer.domElement);

    // 4. Orbit Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Don't flip below ground

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(30, -40, 50);
    scene.add(dirLight);

    // 6. Groups for Unified 2.5D Map Elements
    gridHelper = new THREE.GridHelper(100, 50, 0x00d2d3, 0x17263c);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.z = -1.8;
    scene.add(gridHelper);

    curbsGroup = new THREE.Group();
    potholesGroup = new THREE.Group();
    obstaclesGroup = new THREE.Group();

    scene.add(curbsGroup);
    scene.add(potholesGroup);
    scene.add(obstaclesGroup);

    // 7. Initialize 2.5D Elevation Mesh
    initElevationMesh(60, 60, 80, 80);

    // 8. Event Listeners & UI Controls
    setupUIEventListeners();
    window.addEventListener('resize', onWindowResize);

    // 9. Connect WebSocket Stream
    connectWebSocket();

    // 10. Start Animation Loop
    animate();
}

function initElevationMesh(widthM, heightM, cols, rows) {
    if (elevationMesh) scene.remove(elevationMesh);
    if (elevationPoints) scene.remove(elevationPoints);

    elevationGeometry = new THREE.PlaneGeometry(widthM, heightM, cols - 1, rows - 1);
    
    // Allocate vertex colors buffer
    const count = elevationGeometry.attributes.position.count;
    const colors = new Float32Array(count * 3);
    elevationGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Mesh Material (Double-sided for clean isometric viewing)
    elevationMaterial = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.6,
        metalness: 0.15,
        flatShading: false,
        side: THREE.DoubleSide
    });

    elevationMesh = new THREE.Mesh(elevationGeometry, elevationMaterial);
    scene.add(elevationMesh);

    // Alternative Points Material
    const pointsMat = new THREE.PointsMaterial({
        size: 0.35,
        vertexColors: true,
        transparent: true,
        opacity: 0.9
    });
    elevationPoints = new THREE.Points(elevationGeometry, pointsMat);
    elevationPoints.visible = false;
    scene.add(elevationPoints);
}

function updateElevationMesh(dem) {
    if (!dem || !dem.elevation || dem.elevation.length === 0) return;

    const cols = dem.width;
    const rows = dem.height;
    const bounds = dem.bounds;
    const widthM = bounds[1] - bounds[0];
    const heightM = bounds[3] - bounds[2];

    const currentCols = elevationGeometry.parameters.widthSegments + 1;
    const currentRows = elevationGeometry.parameters.heightSegments + 1;

    // Rebuild geometry only if dimension changed
    if (currentCols !== cols || currentRows !== rows) {
        initElevationMesh(widthM, heightM, cols, rows);
    }

    // Reposition plane center
    const centerX = (bounds[0] + bounds[1]) / 2.0;
    const centerY = (bounds[2] + bounds[3]) / 2.0;
    elevationMesh.position.set(centerX, centerY, 0);
    elevationPoints.position.set(centerX, centerY, 0);

    const positions = elevationGeometry.attributes.position.array;
    const colors = elevationGeometry.attributes.color.array;
    const elev = dem.elevation;
    const minZ = dem.min_z;
    const maxZ = dem.max_z;

    // Update colorbar scale numbers
    document.getElementById('scale-max').innerText = `+${maxZ.toFixed(1)}m`;
    document.getElementById('scale-min').innerText = `${minZ.toFixed(1)}m`;

    for (let i = 0; i < elev.length; i++) {
        const z = elev[i];
        // Set Z coordinate in plane
        positions[i * 3 + 2] = z;

        // Compute vertex color gradient
        const col = getElevationColor(z, minZ, maxZ);
        colors[i * 3] = col.r;
        colors[i * 3 + 1] = col.g;
        colors[i * 3 + 2] = col.b;
    }

    elevationGeometry.attributes.position.needsUpdate = true;
    elevationGeometry.attributes.color.needsUpdate = true;
    elevationGeometry.computeVertexNormals();
}

function updateCurbs(curbPoints) {
    // Clear old curbs
    while (curbsGroup.children.length > 0) {
        const obj = curbsGroup.children.pop();
        if (obj.geometry) obj.geometry.dispose();
    }

    if (!curbPoints || curbPoints.length === 0) return;

    const positions = new Float32Array(curbPoints.length * 3);
    for (let i = 0; i < curbPoints.length; i++) {
        positions[i * 3] = curbPoints[i][0];
        positions[i * 3 + 1] = curbPoints[i][1];
        positions[i * 3 + 2] = curbPoints[i][2] + 0.04; // Slight offset above road
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
        size: 0.35,
        color: 0x00d2d3, // Cyan curb highlight
        transparent: true,
        opacity: 0.95
    });

    const curbCloud = new THREE.Points(geom, mat);
    curbsGroup.add(curbCloud);
}

function updatePotholes(potholes) {
    while (potholesGroup.children.length > 0) {
        const obj = potholesGroup.children.pop();
        if (obj.geometry) obj.geometry.dispose();
    }

    const detailList = document.getElementById('pothole-detail-list');
    detailList.innerHTML = '';

    if (!potholes || potholes.length === 0) {
        detailList.innerHTML = '<div class="pothole-tag" style="color:#8395a7">No active road hazards</div>';
        return;
    }

    potholes.forEach((ph, idx) => {
        const cx = ph.center[0];
        const cy = ph.center[1];
        const cz = ph.center[2];
        const radius = ph.radius;
        const depth = ph.max_depth;

        // 3D Ring on road surface
        const ringGeo = new THREE.RingGeometry(radius - 0.15, radius + 0.15, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0xe056fd,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8
        });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.position.set(cx, cy, cz + 0.05);
        potholesGroup.add(ringMesh);

        // Center Marker Cone (pointing into depression)
        const coneGeo = new THREE.ConeGeometry(0.3, 0.6, 16);
        const coneMat = new THREE.MeshBasicMaterial({ color: 0xe056fd });
        const coneMesh = new THREE.Mesh(coneGeo, coneMat);
        coneMesh.rotation.x = Math.PI; // point downwards
        coneMesh.position.set(cx, cy, cz + 0.3);
        potholesGroup.add(coneMesh);

        // Sidebar detail item
        const tag = document.createElement('div');
        tag.className = 'pothole-tag';
        tag.innerHTML = `<span>Hazard #${idx+1} [${cx.toFixed(1)}, ${cy.toFixed(1)}]</span><span>-${depth.toFixed(2)}m</span>`;
        detailList.appendChild(tag);
    });
}

function updateObstacles(obstacles) {
    while (obstaclesGroup.children.length > 0) {
        const obj = obstaclesGroup.children.pop();
        if (obj.geometry) obj.geometry.dispose();
    }

    let vehicleCount = 0;
    let pedCount = 0;
    let poleCount = 0;

    if (!obstacles) return;

    obstacles.forEach(obs => {
        const cx = obs.position[0];
        const cy = obs.position[1];
        const cz = obs.position[2];
        const dx = obs.size[0];
        const dy = obs.size[1];
        const dz = obs.size[2];

        let boxColor = 0xff6b6b; // vehicle default (warm orange/red)
        if (obs.category === 'pedestrian') {
            boxColor = 0xfeca57; // amber
            pedCount++;
        } else if (obs.category === 'pole' || obs.category === 'tree') {
            boxColor = 0xf1c40f; // yellow
            poleCount++;
        } else {
            vehicleCount++;
        }

        // 3D Wireframe Box
        const boxGeo = new THREE.BoxGeometry(dx, dy, dz);
        const wireframe = new THREE.WireframeGeometry(boxGeo);
        const line = new THREE.LineSegments(wireframe);
        line.material.color.setHex(boxColor);
        line.material.linewidth = 2;
        line.position.set(cx, cy, cz);
        obstaclesGroup.add(line);
    });

    document.getElementById('stat-vehicles').innerText = vehicleCount;
    document.getElementById('stat-pedestrians').innerText = pedCount;
    document.getElementById('stat-poles').innerText = poleCount;
}

function updateTelemetry(telemetry) {
    if (!telemetry) return;
    document.getElementById('val-source').innerText = telemetry.source || 'Synthetic Stream';
    document.getElementById('val-format').innerText = telemetry.format || 'LiDAR';
    document.getElementById('val-raw-points').innerText = `${(telemetry.raw_points || 0).toLocaleString()} pts`;
    document.getElementById('val-processed-points').innerText = `${(telemetry.processed_points || 0).toLocaleString()} pts`;
    
    const lat = telemetry.timings_ms ? telemetry.timings_ms.total_pipeline_ms : 28.5;
    document.getElementById('val-latency').innerText = `${lat.toFixed(1)} ms`;
    document.getElementById('val-fps').innerText = `${(telemetry.fps || liveFps).toFixed(1)} FPS`;
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`[FoveaGrid] Connecting to WebSocket stream: ${wsUrl}`);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[FoveaGrid] WebSocket connection established.');
        const pill = document.getElementById('status-pill');
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        pill.classList.remove('disconnected');
        dot.className = 'dot live';
        txt.innerText = 'Connected: Live 2.5D Stream';
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'unified_25d_frame') {
                updateElevationMesh(data.dem);
                updateCurbs(data.curbs);
                updatePotholes(data.potholes);
                updateObstacles(data.obstacles);
                updateTelemetry(data.telemetry);
                document.getElementById('stat-potholes').innerText = data.potholes ? data.potholes.length : 0;
            }
        } catch (e) {
            console.error('[FoveaGrid] Error parsing frame payload:', e);
        }
    };

    ws.onclose = () => {
        console.warn('[FoveaGrid] WebSocket disconnected. Retrying in 2.5s...');
        const pill = document.getElementById('status-pill');
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        pill.classList.add('disconnected');
        dot.className = 'dot dead';
        txt.innerText = 'Disconnected (Auto-Reconnecting)';
        setTimeout(connectWebSocket, 2500);
    };

    ws.onerror = (err) => {
        console.warn('[FoveaGrid] WebSocket encountered an error.', err);
    };
}

function setCameraPreset(preset) {
    if (!camera) return;

    if (preset === 'isometric') {
        camera.position.set(22, -42, 32);
        camera.up.set(0, 0, 1);
        if (controls) controls.target.set(5, 0, -0.5);
    } else if (preset === 'bev') {
        camera.position.set(5, 0, 65);
        camera.up.set(0, 1, 0);
        if (controls) controls.target.set(5, 0, 0);
    } else if (preset === 'driver') {
        camera.position.set(-18, 0, 1.2);
        camera.up.set(0, 0, 1);
        if (controls) controls.target.set(20, 0, -1.0);
    } else if (preset === 'reset') {
        setCameraPreset('isometric');
    }

    if (controls) controls.update();
}

function setupUIEventListeners() {
    // 1. Play / Pause Button
    const playBtn = document.getElementById('btn-play-pause');
    playBtn.addEventListener('click', () => {
        isPaused = !isPaused;
        if (isPaused) {
            playBtn.classList.remove('active');
            document.getElementById('play-icon').innerText = '▶';
            document.getElementById('play-text').innerText = 'Play';
            if (ws && ws.readyState === WebSocket.OPEN) ws.send('pause');
            fetch('/api/stream/pause', { method: 'POST' }).catch(() => {});
        } else {
            playBtn.classList.add('active');
            document.getElementById('play-icon').innerText = '⏸';
            document.getElementById('play-text').innerText = 'Pause';
            if (ws && ws.readyState === WebSocket.OPEN) ws.send('play');
            fetch('/api/stream/play', { method: 'POST' }).catch(() => {});
        }
    });

    // 2. Step Frame Button
    document.getElementById('btn-step').addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send('step');
        fetch('/api/stream/step', { method: 'POST' }).catch(() => {});
    });

    // 3. Source Dropdown
    document.getElementById('select-source').addEventListener('change', (e) => {
        const val = e.target.value;
        fetch('/api/stream/source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: val })
        }).catch(() => {});
    });

    // 4. Camera Preset Buttons
    document.querySelectorAll('.cam-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            setCameraPreset(btn.getAttribute('data-view'));
        });
    });

    // 5. Mesh Style Buttons (Solid, Wire, Points)
    document.querySelectorAll('.style-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentStyle = btn.getAttribute('data-style');
            
            if (currentStyle === 'surface') {
                elevationMaterial.wireframe = false;
                elevationMesh.visible = true;
                elevationPoints.visible = false;
            } else if (currentStyle === 'wireframe') {
                elevationMaterial.wireframe = true;
                elevationMesh.visible = true;
                elevationPoints.visible = false;
            } else if (currentStyle === 'points') {
                elevationMesh.visible = false;
                elevationPoints.visible = true;
            }
        });
    });

    // 6. Layer Visibility Checkboxes
    document.getElementById('layer-mesh').addEventListener('change', (e) => {
        elevationMesh.visible = e.target.checked && (currentStyle !== 'points');
        elevationPoints.visible = e.target.checked && (currentStyle === 'points');
    });
    document.getElementById('layer-boxes').addEventListener('change', (e) => {
        obstaclesGroup.visible = e.target.checked;
    });
    document.getElementById('layer-curbs').addEventListener('change', (e) => {
        curbsGroup.visible = e.target.checked;
    });
    document.getElementById('layer-potholes').addEventListener('change', (e) => {
        potholesGroup.visible = e.target.checked;
    });
    document.getElementById('layer-grid').addEventListener('change', (e) => {
        gridHelper.visible = e.target.checked;
    });
}

function onWindowResize() {
    const container = document.getElementById('three-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    requestAnimationFrame(animate);

    // Compute live FPS
    fpsCounter++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
        liveFps = (fpsCounter * 1000) / (now - lastFrameTime);
        fpsCounter = 0;
        lastFrameTime = now;
    }

    if (controls) controls.update();
    renderer.render(scene, camera);
}

window.onload = initThree;
