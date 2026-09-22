// FoveaGrid Three.js Visualizer App
let scene, camera, renderer, controls;
let pointCloud, foveaGridGroup, boundingBoxGroup;
let currentView = 'view1';

const viewTitles = {
    'view1': 'View 1: Raw 64-Beam LiDAR Point Cloud (Ring Color-Coded)',
    'view2': 'View 2: Semantic Segmentation (Terrain, Static, Dynamic, Overhead)',
    'view3': 'View 3: FoveaGrid Adaptive Resolution Hierarchy (5cm to 50cm)',
    'view4': 'View 4: Dynamic Object Tracking & Velocity Vectors',
    'view5': 'View 5: Safety Risk Map Layer (Conservative Aggregation)',
    'view6': 'View 6: Real-time Performance & Memory Compression Dashboard'
};

function initThree() {
    const container = document.getElementById('three-canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05080e);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, -35, 25);
    camera.up.set(0, 0, 1);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Grid Floor
    const gridHelper = new THREE.GridHelper(200, 40, 0x38ef7d, 0x1a2638);
    gridHelper.rotation.x = Math.PI / 2;
    scene.add(gridHelper);

    // Groups
    foveaGridGroup = new THREE.Group();
    boundingBoxGroup = new THREE.Group();
    scene.add(foveaGridGroup);
    scene.add(boundingBoxGroup);

    // Generate Synthetic Visual Scan Data
    generateSyntheticVisualData();

    // Tab Event Listeners
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentView = btn.getAttribute('data-view');
            document.getElementById('view-title-label').innerText = viewTitles[currentView];
            updateVisualMode();
        });
    });

    window.addEventListener('resize', onWindowResize);
    animate();
}

function generateSyntheticVisualData() {
    const N = 40000;
    const positions = new Float32Array(N * 3);
    const colors = new Float32Array(N * 3);

    const terrainColor = new THREE.Color(0x2ecc71);
    const staticColor  = new THREE.Color(0x95a5a6);
    const dynamicColor = new THREE.Color(0xe74c3c);
    const overheadColor= new THREE.Color(0xf1c40f);
    const potholeColor = new THREE.Color(0x9b59b6);

    for (let i = 0; i < N; i++) {
        const r = Math.random() * 80.0 + 0.5;
        const theta = Math.random() * Math.PI * 2.0;

        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta);
        let z = 0.0;
        let col = terrainColor;

        const randVal = Math.random();
        if (randVal > 0.85) {
            // Static obstacle
            z = Math.random() * 3.0 + 0.5;
            col = staticColor;
        } else if (randVal > 0.78) {
            // Dynamic object (Pedestrian/Car at 35m)
            z = Math.random() * 1.6 + 0.2;
            col = dynamicColor;
        } else if (randVal > 0.73 && Math.abs(x - 30) < 5) {
            // Overhead obstacle
            z = 4.2 + Math.random() * 0.8;
            col = overheadColor;
        } else if (Math.hypot(x - 25, y - 1.2) < 1.2) {
            // Pothole
            z = -0.2;
            col = potholeColor;
        } else {
            // Terrain ground
            z = Math.sin(x * 0.1) * 0.1;
        }

        positions[i*3] = x;
        positions[i*3+1] = y;
        positions[i*3+2] = z;

        colors[i*3] = col.r;
        colors[i*3+1] = col.g;
        colors[i*3+2] = col.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.25,
        vertexColors: true,
        transparent: true,
        opacity: 0.85
    });

    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);

    // Build FoveaGrid Visual Rings
    createFoveaGridRings();
}

function createFoveaGridRings() {
    const rings = [
        { r: 10, res: '5 cm', col: 0x38ef7d },
        { r: 25, res: '10 cm', col: 0x11998e },
        { r: 50, res: '20 cm', col: 0x3498db },
        { r: 75, res: '35 cm', col: 0x9b59b6 },
        { r: 100, res: '50 cm', col: 0xe67e22 }
    ];

    rings.forEach(ring => {
        const ringGeo = new THREE.RingGeometry(ring.r - 0.2, ring.r + 0.2, 64);
        const ringMat = new THREE.MeshBasicMaterial({ color: ring.col, side: THREE.DoubleSide, transparent: true, opacity: 0.6 });
        const mesh = new THREE.Mesh(ringGeo, ringMat);
        foveaGridGroup.add(mesh);
    });
}

function updateVisualMode() {
    if (currentView === 'view3' || currentView === 'view5') {
        foveaGridGroup.visible = true;
    } else {
        foveaGridGroup.visible = false;
    }
}

function onWindowResize() {
    const container = document.getElementById('three-canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

window.onload = initThree;
