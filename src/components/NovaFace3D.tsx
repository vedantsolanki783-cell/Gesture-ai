import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface Props {
  className?: string;
}

export function NovaFace3D({ className = '' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    let isUnmounted = false;

    // 1. Core Scene Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, containerRef.current.clientWidth / containerRef.current.clientHeight, 0.1, 100);
    camera.position.z = 4.5;

    // Transparent renderer to seamlessly blend with your Cyber-Glass background
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap at 2x for mobile performance
    containerRef.current.appendChild(renderer.domElement);

    const headGroup = new THREE.Group();
    scene.add(headGroup);

    // 2. Head Geometry (Glossy Obsidian Sphere)
    const headGeo = new THREE.SphereGeometry(1, 64, 64);
    headGeo.scale(1, 1.05, 0.95); // Slightly egg-shaped/squished
    const headMat = new THREE.MeshPhysicalMaterial({
      color: 0x050711,
      metalness: 0.9,
      roughness: 0.15,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
    });
    const headMesh = new THREE.Mesh(headGeo, headMat);
    headGroup.add(headMesh);

    // 3. Eyes (Glowing Horizontal Capsules)
    const eyeGeo = new THREE.CapsuleGeometry(0.08, 0.22, 16, 16);
    eyeGeo.rotateZ(Math.PI / 2); // Lay pill horizontally
    const eyeMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 }); // Neon cyan
    
    const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
    eyeL.position.set(-0.35, 0.05, 0.94);
    const eyeLightL = new THREE.PointLight(0x38bdf8, 1.5, 2);
    eyeL.add(eyeLightL);
    
    const eyeR = new THREE.Mesh(eyeGeo, eyeMat);
    eyeR.position.set(0.35, 0.05, 0.94);
    const eyeLightR = new THREE.PointLight(0x38bdf8, 1.5, 2);
    eyeR.add(eyeLightR);

    const eyesGroup = new THREE.Group();
    eyesGroup.add(eyeL, eyeR);
    headGroup.add(eyesGroup);

    // 4. Energy Vortex (Swirling Holographic Rings)
    const auraGroup = new THREE.Group();
    scene.add(auraGroup);
    
    const createRing = (radius: number, color: number, rx: number, ry: number) => {
      const geo = new THREE.TorusGeometry(radius, 0.015, 16, 100);
      const mat = new THREE.MeshBasicMaterial({ 
        color, 
        transparent: true, 
        opacity: 0.6, 
        blending: THREE.AdditiveBlending 
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.set(rx, ry, 0);
      return mesh;
    };
    
    // Magenta, Purple, and Cyan energy layers
    const ring1 = createRing(1.3, 0x38bdf8, Math.PI / 3, Math.PI / 6);
    const ring2 = createRing(1.4, 0xc084fc, -Math.PI / 4, Math.PI / 3);
    const ring3 = createRing(1.5, 0xec4899, Math.PI / 2, -Math.PI / 4);
    const ring4 = createRing(1.25, 0x00f0ff, Math.PI / 6, -Math.PI / 2);
    auraGroup.add(ring1, ring2, ring3, ring4);

    // 5. Studio Lighting Setup
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const cyanLight = new THREE.DirectionalLight(0x38bdf8, 3.5);
    cyanLight.position.set(-3, 3, 2);
    scene.add(cyanLight); // Top-left cyan rim light
    const purpleLight = new THREE.DirectionalLight(0xc084fc, 3.5);
    purpleLight.position.set(3, -3, 2);
    scene.add(purpleLight); // Bottom-right purple rim light

    // 6. Interactive Spatial Kinematics (Look-At)
    const targetQuaternion = new THREE.Quaternion();
    const dummyObj = new THREE.Object3D();
    let lookTimer = 0;
    
    const onGlobalClick = (e: MouseEvent) => {
      // Map screen click coordinates to WebGL normalized device coordinates
      const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
      const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;
      
      // Project the click point into 3D space toward the viewer (+Z)
      const targetPos = new THREE.Vector3(ndcX * 6, ndcY * 6, 5);
      
      // Calculate the exact 3D angle required to face the click
      dummyObj.position.copy(headGroup.position);
      dummyObj.lookAt(targetPos);
      targetQuaternion.copy(dummyObj.quaternion);
      
      lookTimer = performance.now() + 2500; // Hold the gaze for 2.5 seconds
      
      // Trigger attentive expression (squint)
      eyesGroup.scale.y = 0.2;
      setTimeout(() => { if (!isUnmounted) eyesGroup.scale.y = 1.0; }, 150);
    };
    window.addEventListener('click', onGlobalClick);

    const onResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', onResize);

    // 7. Hardware Animation Loop
    let frameId: number;
    let nextBlinkTime = performance.now() + 3000;

    const animate = (time: number) => {
      // Rotate holographic rings independently
      ring1.rotation.x += 0.005; ring1.rotation.y += 0.002;
      ring2.rotation.x -= 0.003; ring2.rotation.y += 0.004;
      ring3.rotation.x += 0.004; ring3.rotation.y -= 0.005;
      ring4.rotation.x -= 0.002; ring4.rotation.y -= 0.003;

      // Organic hovering animation
      headGroup.position.y = Math.sin(time * 0.002) * 0.06;
      auraGroup.position.y = Math.sin(time * 0.002) * 0.06;

      // Autonomous random blinking
      if (time > nextBlinkTime) {
        eyesGroup.scale.y = 0.1; // Squish eyes flat
        setTimeout(() => { if (!isUnmounted) eyesGroup.scale.y = 1.0; }, 120);
        nextBlinkTime = time + Math.random() * 4000 + 2000; // Blink every 2-6 seconds
      }

      // Smooth kinematic rotation (Slerp)
      if (time < lookTimer) {
        // Snap gaze quickly toward the mouse click
        headGroup.quaternion.slerp(targetQuaternion, 0.08);
      } else {
        // Idle state: Slowly pan around generating an "alive" searching look
        const idleTarget = new THREE.Vector3(
          Math.sin(time * 0.001) * 0.5, 
          Math.cos(time * 0.0013) * 0.2, 
          5
        );
        dummyObj.position.copy(headGroup.position);
        dummyObj.lookAt(idleTarget);
        targetQuaternion.copy(dummyObj.quaternion);
        headGroup.quaternion.slerp(targetQuaternion, 0.02); // Drift slowly
      }

      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    frameId = requestAnimationFrame(animate);

    return () => {
      isUnmounted = true;
      window.removeEventListener('click', onGlobalClick);
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(frameId);
      containerRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  return <div ref={containerRef} className={`nova-3d-container ${className}`} style={{ width: '100%', height: '100%', minHeight: '300px' }} />;
}
