export type Kind = 'planar_face' | 'hole' | 'pocket' | 'slot'
export type Machine = 'cnc_mill' | 'drill_press'
export type Tool = 'face_mill' | 'spot_drill' | 'twist_drill' | 'end_mill' | 'slot_mill'
export interface Feature {
  id: string; kind: Kind; quantity: number; diameter_mm: number;
  depth_mm: number; length_mm: number; width_mm: number
}
export interface Request {
  part_name: string; material: 'aluminum' | 'steel'; features: Feature[];
  available_machines: Machine[]; available_tools: Tool[]
}
export interface Operation {
  id: string; feature_id: string; action: string; machine: Machine;
  tool: Tool; process_minutes: number; predecessors: string[]; note: string
}
export interface Summary {
  process_minutes: number; changeover_minutes: number;
  total_estimated_minutes: number; tool_changes: number; machine_changes: number;
  operation_count: number; method: string
}
export interface Route { operations: Operation[]; summary: Summary }
export interface Response {
  part_name: string; material: string; baseline: Route;
  optimized: Route; improvement_percent: number; scope_note: string
}
export interface GeometryResponse {
  solid_count: number; face_count: number; cylindrical_face_count: number;
  bounding_box_mm: {x:number;y:number;z:number}; volume_mm3:number;
  classification_note: string
}
