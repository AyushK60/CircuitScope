export const CONTENT_W = 780;
// model level is the tallest; explicit rows guarantee correct ordering
export const MODEL_H = 1780;
export const BLOCK_Y0 = 330;
export const BLOCK_PITCH = 90;
export const BLOCK_H = 78;

export interface Cam {
  x: number;
  y: number;
  k: number;
}

export const VIEW_W = 900;
export const VIEW_H = 800;

export function fitCam(): Cam {
  const k = Math.min(VIEW_W / (CONTENT_W + 80), VIEW_H / (MODEL_H + 60));
  return { x: (VIEW_W - CONTENT_W * k) / 2 + 10, y: 8, k };
}
