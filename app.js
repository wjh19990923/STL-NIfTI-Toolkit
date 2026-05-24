import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const state = {
  downloadBlob: null,
  downloadName: "",
  scene: null,
  renderer: null,
  camera: null,
  controls: null,
  meshObject: null,
};

const $ = (id) => document.getElementById(id);

function setStatus(message) {
  $("status-text").textContent = message;
}

function logResult(value) {
  $("result-log").textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setDownload(blob, name) {
  state.downloadBlob = blob;
  state.downloadName = name;
  $("download-button").disabled = !blob;
}

function activeFile(inputId) {
  const file = $(inputId).files?.[0];
  if (!file) throw new Error("Please choose an input file first.");
  return file;
}

async function readFileBytes(file) {
  const buffer = await file.arrayBuffer();
  if (file.name.toLowerCase().endsWith(".gz")) {
    if (!("DecompressionStream" in window)) {
      throw new Error("This browser cannot decompress .nii.gz. Please use an uncompressed .nii file.");
    }
    const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return new Uint8Array(buffer);
}

function initViewer() {
  if (state.renderer) return;
  const container = $("viewer");
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x101820);
  state.camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100000);
  state.camera.position.set(130, 100, 160);

  state.renderer = new THREE.WebGLRenderer({ antialias: true });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(state.renderer.domElement);

  state.controls = new OrbitControls(state.camera, state.renderer.domElement);
  state.controls.enableDamping = true;

  const ambient = new THREE.HemisphereLight(0xffffff, 0x263238, 2.2);
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(90, 120, 70);
  state.scene.add(ambient, key, new THREE.GridHelper(240, 12, 0x4b5563, 0x27313c));

  const animate = () => {
    requestAnimationFrame(animate);
    state.controls.update();
    state.renderer.render(state.scene, state.camera);
  };
  animate();

  window.addEventListener("resize", () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    state.camera.aspect = width / height;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(width, height);
  });
}

function previewMesh(mesh, color = 0x2dd4bf) {
  initViewer();
  if (state.meshObject) {
    state.scene.remove(state.meshObject);
    state.meshObject.geometry.dispose();
    state.meshObject.material.dispose();
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3));
  geometry.setIndex(mesh.faces.flat());
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.05 });
  state.meshObject = new THREE.Mesh(geometry, material);
  state.scene.add(state.meshObject);

  const box = new THREE.Box3().setFromObject(state.meshObject);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) || 100;
  state.controls.target.copy(center);
  state.camera.position.set(center.x + radius * 1.4, center.y + radius * 1.1, center.z + radius * 1.5);
  state.camera.near = Math.max(radius / 1000, 0.01);
  state.camera.far = radius * 20;
  state.camera.updateProjectionMatrix();
  $("viewer-caption").textContent = `${mesh.vertices.length.toLocaleString()} vertices, ${mesh.faces.length.toLocaleString()} faces`;
}

function previewNiftiVoxels(nifti) {
  const threshold = estimateNiftiThreshold(nifti);
  const mesh = niftiToPreviewVoxelMesh(nifti, threshold, 7000);
  previewMesh(mesh, 0xf59e0b);
  $("viewer-caption").textContent = `NIfTI voxel preview: ${mesh.previewVoxels.toLocaleString()} sampled cubes from ${mesh.activeVoxels.toLocaleString()} active voxels`;
  return {
    threshold,
    activeVoxels: mesh.activeVoxels,
    previewVoxels: mesh.previewVoxels,
    samplingStep: mesh.samplingStep,
    faces: mesh.faces.length,
  };
}

function estimateNiftiThreshold(nifti) {
  const [nx, ny, nz] = nifti.shape;
  const total = nx * ny * nz;
  const stride = Math.max(1, Math.floor(total / 20000));
  const values = [];
  for (let idx = 0; idx < total; idx += stride) {
    const value = niftiValueAt(nifti, idx);
    if (Number.isFinite(value)) values.push(value);
  }
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  const min = values[0];
  const max = values[values.length - 1];
  if (min < -500 && max > 500) return 300;
  if (min <= 0 && max <= 1) return 0.5;
  if (min < 0 && max > 0) return 0;
  return values[Math.floor(values.length * 0.75)] ?? min;
}

function niftiToPreviewVoxelMesh(nifti, threshold, maxPreviewVoxels) {
  const [nx, ny, nz] = nifti.shape;
  const total = nx * ny * nz;
  let activeVoxels = 0;
  for (let idx = 0; idx < total; idx += 1) {
    if (niftiValueAt(nifti, idx) > threshold) activeVoxels += 1;
  }

  if (!activeVoxels) {
    throw new Error(`No voxels passed the preview threshold ${threshold.toFixed(2)}.`);
  }

  const samplingStep = Math.max(1, Math.ceil(Math.cbrt(activeVoxels / maxPreviewVoxels)));
  const vertices = [];
  const faces = [];
  const spacing = [
    Math.abs(nifti.pixdim[1]) || 1,
    Math.abs(nifti.pixdim[2]) || 1,
    Math.abs(nifti.pixdim[3]) || 1,
  ];
  const origin = [nifti.affine[0][3] || 0, nifti.affine[1][3] || 0, nifti.affine[2][3] || 0];
  const cubeSize = spacing.map((value) => value * Math.max(1, samplingStep) * 0.82);
  const half = cubeSize.map((value) => value / 2);

  for (let k = 0; k < nz; k += samplingStep) {
    for (let j = 0; j < ny; j += samplingStep) {
      for (let i = 0; i < nx; i += samplingStep) {
        const idx = i + nx * (j + ny * k);
        if (niftiValueAt(nifti, idx) <= threshold) continue;
        const center = [
          origin[0] + (i + 0.5) * spacing[0],
          origin[1] + (j + 0.5) * spacing[1],
          origin[2] + (k + 0.5) * spacing[2],
        ];
        addPreviewCube(vertices, faces, center, half);
        if (faces.length / 12 >= maxPreviewVoxels) {
          return { vertices, faces, activeVoxels, previewVoxels: faces.length / 12, samplingStep };
        }
      }
    }
  }

  return { vertices, faces, activeVoxels, previewVoxels: faces.length / 12, samplingStep };
}

function addPreviewCube(vertices, faces, center, half) {
  const [cx, cy, cz] = center;
  const [hx, hy, hz] = half;
  const base = vertices.length;
  vertices.push(
    [cx - hx, cy - hy, cz - hz],
    [cx + hx, cy - hy, cz - hz],
    [cx + hx, cy + hy, cz - hz],
    [cx - hx, cy + hy, cz - hz],
    [cx - hx, cy - hy, cz + hz],
    [cx + hx, cy - hy, cz + hz],
    [cx + hx, cy + hy, cz + hz],
    [cx - hx, cy + hy, cz + hz],
  );
  faces.push(
    [base, base + 2, base + 1],
    [base, base + 3, base + 2],
    [base + 4, base + 5, base + 6],
    [base + 4, base + 6, base + 7],
    [base, base + 1, base + 5],
    [base, base + 5, base + 4],
    [base + 1, base + 2, base + 6],
    [base + 1, base + 6, base + 5],
    [base + 2, base + 3, base + 7],
    [base + 2, base + 7, base + 6],
    [base + 3, base, base + 4],
    [base + 3, base + 4, base + 7],
  );
}

function parseStl(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const triangleCount = view.getUint32(80, true);
  const expectedBinarySize = 84 + triangleCount * 50;
  if (bytes.byteLength >= 84 && expectedBinarySize === bytes.byteLength) {
    return parseBinaryStl(view, triangleCount);
  }
  const text = new TextDecoder().decode(bytes);
  return parseAsciiStl(text);
}

function parseBinaryStl(view, triangleCount) {
  const vertices = [];
  const faces = [];
  let offset = 84;
  for (let i = 0; i < triangleCount; i += 1) {
    offset += 12;
    const face = [];
    for (let j = 0; j < 3; j += 1) {
      vertices.push([
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ]);
      face.push(vertices.length - 1);
      offset += 12;
    }
    faces.push(face);
    offset += 2;
  }
  return { vertices, faces };
}

function parseAsciiStl(text) {
  const vertices = [];
  const faces = [];
  const vertexPattern = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;
  let match;
  while ((match = vertexPattern.exec(text))) {
    vertices.push([Number(match[1]), Number(match[2]), Number(match[3])]);
    if (vertices.length % 3 === 0) {
      const n = vertices.length;
      faces.push([n - 3, n - 2, n - 1]);
    }
  }
  if (!faces.length) throw new Error("No triangles were found in this STL file.");
  return { vertices, faces };
}

function meshBounds(mesh) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const v of mesh.vertices) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], v[axis]);
      max[axis] = Math.max(max[axis], v[axis]);
    }
  }
  return { min, max, extent: max.map((value, i) => value - min[i]) };
}

function parseNifti(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const sizeofHdr = view.getInt32(0, true);
  if (sizeofHdr !== 348) throw new Error("Invalid NIfTI-1 header.");
  const dims = [];
  for (let i = 0; i < 8; i += 1) dims.push(view.getInt16(40 + i * 2, true));
  const datatype = view.getInt16(70, true);
  const bitpix = view.getInt16(72, true);
  const pixdim = [];
  for (let i = 0; i < 8; i += 1) pixdim.push(view.getFloat32(76 + i * 4, true));
  const voxOffset = Math.floor(view.getFloat32(108, true));
  const affine = [
    [view.getFloat32(280, true), view.getFloat32(284, true), view.getFloat32(288, true), view.getFloat32(292, true)],
    [view.getFloat32(296, true), view.getFloat32(300, true), view.getFloat32(304, true), view.getFloat32(308, true)],
    [view.getFloat32(312, true), view.getFloat32(316, true), view.getFloat32(320, true), view.getFloat32(324, true)],
    [0, 0, 0, 1],
  ];
  const shape = [dims[1], dims[2], dims[3]].map((v) => Math.max(1, v));
  return { bytes, shape, datatype, bitpix, pixdim, voxOffset, affine };
}

function niftiValueAt(nifti, index) {
  const view = new DataView(nifti.bytes.buffer, nifti.bytes.byteOffset + nifti.voxOffset);
  const offset = index * (nifti.bitpix / 8);
  switch (nifti.datatype) {
    case 2:
      return view.getUint8(offset);
    case 4:
      return view.getInt16(offset, true);
    case 8:
      return view.getInt32(offset, true);
    case 16:
      return view.getFloat32(offset, true);
    case 64:
      return view.getFloat64(offset, true);
    case 512:
      return view.getUint16(offset, true);
    default:
      throw new Error(`Unsupported NIfTI datatype: ${nifti.datatype}`);
  }
}

function createNiftiFromStl(mesh, voxelSize, hu, fillInterior) {
  const bounds = meshBounds(mesh);
  const padding = voxelSize * 2;
  const origin = bounds.min.map((v) => v - padding);
  const dims = bounds.extent.map((v) => Math.max(3, Math.ceil((v + padding * 2) / voxelSize) + 1));
  const voxelCount = dims[0] * dims[1] * dims[2];
  if (voxelCount > 9000000) {
    throw new Error(`Requested grid has ${voxelCount.toLocaleString()} voxels. Increase voxel size for browser processing.`);
  }

  const mask = new Uint8Array(voxelCount);
  const mark = (point) => {
    const i = Math.floor((point[0] - origin[0]) / voxelSize);
    const j = Math.floor((point[1] - origin[1]) / voxelSize);
    const k = Math.floor((point[2] - origin[2]) / voxelSize);
    if (i >= 0 && j >= 0 && k >= 0 && i < dims[0] && j < dims[1] && k < dims[2]) {
      mask[i + dims[0] * (j + dims[1] * k)] = 1;
    }
  };

  for (const face of mesh.faces) {
    const a = mesh.vertices[face[0]];
    const b = mesh.vertices[face[1]];
    const c = mesh.vertices[face[2]];
    const edgeMax = Math.max(distance(a, b), distance(b, c), distance(c, a));
    const steps = Math.max(1, Math.ceil(edgeMax / Math.max(voxelSize * 0.5, 0.1)));
    for (let u = 0; u <= steps; u += 1) {
      for (let v = 0; v <= steps - u; v += 1) {
        const w = steps - u - v;
        const point = [
          (a[0] * u + b[0] * v + c[0] * w) / steps,
          (a[1] * u + b[1] * v + c[1] * w) / steps,
          (a[2] * u + b[2] * v + c[2] * w) / steps,
        ];
        mark(point);
      }
    }
  }

  if (fillInterior) {
    for (let i = 0; i < dims[0]; i += 1) {
      for (let j = 0; j < dims[1]; j += 1) {
        let first = -1;
        let last = -1;
        for (let k = 0; k < dims[2]; k += 1) {
          if (mask[i + dims[0] * (j + dims[1] * k)]) {
            if (first < 0) first = k;
            last = k;
          }
        }
        if (first >= 0 && last > first) {
          for (let k = first; k <= last; k += 1) mask[i + dims[0] * (j + dims[1] * k)] = 1;
        }
      }
    }
  }

  const headerSize = 352;
  const output = new ArrayBuffer(headerSize + voxelCount * 2);
  const view = new DataView(output);
  view.setInt32(0, 348, true);
  view.setInt16(40, 3, true);
  view.setInt16(42, dims[0], true);
  view.setInt16(44, dims[1], true);
  view.setInt16(46, dims[2], true);
  view.setInt16(70, 4, true);
  view.setInt16(72, 16, true);
  view.setFloat32(76, 1, true);
  view.setFloat32(80, voxelSize, true);
  view.setFloat32(84, voxelSize, true);
  view.setFloat32(88, voxelSize, true);
  view.setFloat32(108, headerSize, true);
  view.setFloat32(112, 1, true);
  view.setInt16(252, 2, true);
  view.setFloat32(280, voxelSize, true);
  view.setFloat32(296, voxelSize, true);
  view.setFloat32(312, voxelSize, true);
  view.setFloat32(292, origin[0], true);
  view.setFloat32(308, origin[1], true);
  view.setFloat32(324, origin[2], true);
  new Uint8Array(output, 344, 4).set([110, 43, 49, 0]);
  for (let idx = 0; idx < voxelCount; idx += 1) {
    view.setInt16(headerSize + idx * 2, mask[idx] ? hu : -1024, true);
  }
  return { blob: new Blob([output], { type: "application/octet-stream" }), dims, voxelCount };
}

function niftiToBlockMesh(nifti, threshold, maxVoxels) {
  const [nx, ny, nz] = nifti.shape;
  const selected = [];
  const active = new Set();
  const total = nx * ny * nz;
  for (let idx = 0; idx < total; idx += 1) {
    if (niftiValueAt(nifti, idx) > threshold) {
      active.add(idx);
      selected.push(idx);
      if (selected.length > maxVoxels) {
        throw new Error(`More than ${maxVoxels.toLocaleString()} voxels passed the threshold. Raise threshold or max voxels carefully.`);
      }
    }
  }

  const vertices = [];
  const faces = [];
  const spacing = [nifti.pixdim[1] || 1, nifti.pixdim[2] || 1, nifti.pixdim[3] || 1];
  const origin = [nifti.affine[0][3] || 0, nifti.affine[1][3] || 0, nifti.affine[2][3] || 0];
  const directions = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];
  const faceCorners = [
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
  ];

  for (const idx of selected) {
    const i = idx % nx;
    const j = Math.floor(idx / nx) % ny;
    const k = Math.floor(idx / (nx * ny));
    for (let side = 0; side < 6; side += 1) {
      const ni = i + directions[side][0];
      const nj = j + directions[side][1];
      const nk = k + directions[side][2];
      const neighbor = ni + nx * (nj + ny * nk);
      if (ni >= 0 && nj >= 0 && nk >= 0 && ni < nx && nj < ny && nk < nz && active.has(neighbor)) continue;
      const base = vertices.length;
      for (const corner of faceCorners[side]) {
        vertices.push([
          origin[0] + (i + corner[0]) * spacing[0],
          origin[1] + (j + corner[1]) * spacing[1],
          origin[2] + (k + corner[2]) * spacing[2],
        ]);
      }
      faces.push([base, base + 1, base + 2], [base, base + 2, base + 3]);
    }
  }
  return { vertices, faces, selectedCount: selected.length };
}

function meshToBinaryStl(mesh, name = "created by STL NIfTI Toolkit") {
  const buffer = new ArrayBuffer(84 + mesh.faces.length * 50);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  bytes.set(new TextEncoder().encode(name.slice(0, 80)));
  view.setUint32(80, mesh.faces.length, true);
  let offset = 84;
  for (const face of mesh.faces) {
    const normal = faceNormal(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
    for (const value of normal) {
      view.setFloat32(offset, value, true);
      offset += 4;
    }
    for (const vertexIndex of face) {
      for (const value of mesh.vertices[vertexIndex]) {
        view.setFloat32(offset, value, true);
        offset += 4;
      }
    }
    view.setUint16(offset, 0, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "model/stl" });
}

function samplePoints(mesh, count) {
  const step = Math.max(1, Math.floor(mesh.vertices.length / count));
  const points = [];
  for (let i = 0; i < mesh.vertices.length && points.length < count; i += step) points.push([...mesh.vertices[i]]);
  return points;
}

function estimateTransform(sourceMesh, targetMesh, count, iterations) {
  let source = samplePoints(sourceMesh, count);
  const original = source.map((p) => [...p]);
  const target = samplePoints(targetMesh, count);
  let matrix = identity4();
  for (let iter = 0; iter < iterations; iter += 1) {
    const pairs = source.map((point) => [point, nearestPoint(point, target)]);
    const step = kabsch(pairs.map((p) => p[0]), pairs.map((p) => p[1]));
    source = source.map((point) => transformPoint(step, point));
    matrix = multiply4(step, matrix);
  }
  const residual = source.reduce((sum, point, idx) => sum + distance(point, nearestPoint(point, target)), 0) / source.length;
  return { matrix, residual, sourcePointCount: original.length, targetPointCount: target.length };
}

function nearestPoint(point, candidates) {
  let best = candidates[0];
  let bestD = Infinity;
  for (const candidate of candidates) {
    const d = squaredDistance(point, candidate);
    if (d < bestD) {
      bestD = d;
      best = candidate;
    }
  }
  return best;
}

function kabsch(source, target) {
  const sourceCenter = centroid(source);
  const targetCenter = centroid(target);
  const src = source.map((p) => subtract(p, sourceCenter));
  const dst = target.map((p) => subtract(p, targetCenter));
  const h = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < src.length; i += 1) {
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) h[r][c] += src[i][r] * dst[i][c];
    }
  }
  const q = dominantEigenvector(buildQuaternionMatrix(h));
  const rotation = quaternionToMatrix(q);
  const rotatedSourceCenter = multiplyMatVec(rotation, sourceCenter);
  const translation = subtract(targetCenter, rotatedSourceCenter);
  return [
    [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
    [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
    [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
    [0, 0, 0, 1],
  ];
}

function buildQuaternionMatrix(h) {
  const sxx = h[0][0], sxy = h[0][1], sxz = h[0][2];
  const syx = h[1][0], syy = h[1][1], syz = h[1][2];
  const szx = h[2][0], szy = h[2][1], szz = h[2][2];
  return [
    [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
    [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
    [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
    [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
  ];
}

function dominantEigenvector(matrix) {
  let vector = [1, 0, 0, 0];
  for (let iter = 0; iter < 50; iter += 1) {
    const next = matrix.map((row) => row.reduce((sum, value, i) => sum + value * vector[i], 0));
    const norm = Math.hypot(...next) || 1;
    vector = next.map((v) => v / norm);
  }
  return vector;
}

function quaternionToMatrix(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
  ];
}

function identity4() {
  return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
}

function multiply4(a, b) {
  return a.map((row, r) => row.map((_, c) => row.reduce((sum, value, k) => sum + value * b[k][c], 0)));
}

function transformPoint(matrix, point) {
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
    matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
    matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3],
  ];
}

function centroid(points) {
  const sum = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]], [0, 0, 0]);
  return sum.map((v) => v / points.length);
}

function distance(a, b) {
  return Math.sqrt(squaredDistance(a, b));
}

function squaredDistance(a, b) {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function multiplyMatVec(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, value, i) => sum + value * vector[i], 0));
}

function faceNormal(a, b, c) {
  const u = subtract(b, a);
  const v = subtract(c, a);
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const length = Math.hypot(...n) || 1;
  return n.map((value) => value / length);
}

function fileStem(name) {
  return name.replace(/\.nii\.gz$/i, "").replace(/\.[^.]+$/i, "");
}

function wireEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((node) => node.classList.toggle("active", node === button));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      $(`${button.dataset.tab}-panel`).classList.add("active");
    });
  });

  $("download-button").addEventListener("click", () => {
    if (!state.downloadBlob) return;
    const url = URL.createObjectURL(state.downloadBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = state.downloadName;
    link.click();
    URL.revokeObjectURL(url);
  });

  $("inspect-button").addEventListener("click", async () => {
    try {
      const file = activeFile("inspect-file");
      setStatus(`Reading ${file.name}...`);
      const bytes = await readFileBytes(file);
      if (file.name.toLowerCase().includes(".nii")) {
        const nifti = parseNifti(bytes);
        const preview = previewNiftiVoxels(nifti);
        setDownload(null, "");
        logResult({
          file: file.name,
          shape: nifti.shape,
          datatype: nifti.datatype,
          bitpix: nifti.bitpix,
          voxelSize: nifti.pixdim.slice(1, 4),
          affine: nifti.affine,
          preview,
        });
      } else {
        const mesh = parseStl(bytes);
        previewMesh(mesh);
        const bounds = meshBounds(mesh);
        logResult({ file: file.name, vertices: mesh.vertices.length, faces: mesh.faces.length, bounds });
      }
      setStatus("Inspection complete.");
    } catch (error) {
      setStatus(error.message);
      logResult(error.stack || error.message);
    }
  });

  $("stl-to-nii-button").addEventListener("click", async () => {
    try {
      const file = activeFile("voxel-stl-file");
      const voxelSize = Number($("voxel-size").value);
      const hu = Number($("voxel-hu").value);
      setStatus(`Voxelizing ${file.name}...`);
      const mesh = parseStl(await readFileBytes(file));
      previewMesh(mesh);
      const result = createNiftiFromStl(mesh, voxelSize, hu, $("voxel-fill").checked);
      setDownload(result.blob, `${fileStem(file.name)}_HU${hu}_browser.nii`);
      logResult({ output: state.downloadName, dimensions: result.dims, voxelCount: result.voxelCount, voxelSize, hu });
      setStatus("NIfTI file is ready for download.");
    } catch (error) {
      setStatus(error.message);
      logResult(error.stack || error.message);
    }
  });

  $("nii-to-stl-button").addEventListener("click", async () => {
    try {
      const file = activeFile("nii-file");
      const threshold = Number($("nii-threshold").value);
      const maxVoxels = Number($("nii-max-voxels").value);
      setStatus(`Extracting block surface from ${file.name}...`);
      const nifti = parseNifti(await readFileBytes(file));
      const mesh = niftiToBlockMesh(nifti, threshold, maxVoxels);
      previewMesh(mesh, 0xf59e0b);
      const blob = meshToBinaryStl(mesh, "NIfTI block surface");
      setDownload(blob, `${fileStem(file.name)}_threshold_${threshold}.stl`);
      logResult({ output: state.downloadName, activeVoxels: mesh.selectedCount, faces: mesh.faces.length, threshold });
      setStatus("STL file is ready for download.");
    } catch (error) {
      setStatus(error.message);
      logResult(error.stack || error.message);
    }
  });

  $("match-button").addEventListener("click", async () => {
    try {
      const sourceFile = activeFile("match-source-file");
      const targetFile = activeFile("match-target-file");
      const points = Number($("match-points").value);
      const iterations = Number($("match-iterations").value);
      setStatus("Running browser ICP...");
      const sourceMesh = parseStl(await readFileBytes(sourceFile));
      const targetMesh = parseStl(await readFileBytes(targetFile));
      previewMesh(targetMesh, 0x60a5fa);
      const result = estimateTransform(sourceMesh, targetMesh, points, iterations);
      setDownload(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }), `${fileStem(sourceFile.name)}_to_${fileStem(targetFile.name)}_transform.json`);
      logResult(result);
      setStatus("Transform estimate is ready.");
    } catch (error) {
      setStatus(error.message);
      logResult(error.stack || error.message);
    }
  });
}

initViewer();
wireEvents();
