import { useEffect, useState } from 'react'
import { ArrowRight, CircuitBoard, Download, Factory, FileUp, Gauge, Layers3, Plus, RotateCcw, Trash2 } from 'lucide-react'
import { api } from './api'
import type {Feature, Kind, Machine, Tool, Request, Response, GeometryResponse, Route} from './types'

const defaultFeatures: Feature[] = [
  {id:'face-1', kind:'planar_face', quantity:1, diameter_mm:8, depth_mm:2, length_mm:90, width_mm:55},
  {id:'hole-1', kind:'hole', quantity:4, diameter_mm:8, depth_mm:15, length_mm:10, width_mm:10},
  {id:'pocket-1', kind:'pocket', quantity:1, diameter_mm:8, depth_mm:7, length_mm:40, width_mm:25},
  {id:'slot-1', kind:'slot', quantity:1, diameter_mm:8, depth_mm:4, length_mm:36, width_mm:8}
]
const allTools: Tool[] = ['face_mill','spot_drill','twist_drill','end_mill','slot_mill']
const pretty = (s:string) => s.replaceAll('_',' ').replaceAll('-', ' ')

function RouteTable({title,route}: {title:string; route:Route}) {
  return <section className="route"><div className="routeHead"><h3>{title}</h3><span>{route.summary.method}</span></div>
    <div className="tableScroll"><table><thead><tr><th>#</th><th>Feature</th><th>Operation</th><th>Machine</th><th>Tool</th><th>Time</th></tr></thead>
    <tbody>{route.operations.map((op,i)=><tr key={op.id}><td>{String(i+1).padStart(2,'0')}</td><td>{op.feature_id}</td><td>{pretty(op.action)}</td><td>{pretty(op.machine)}</td><td>{pretty(op.tool)}</td><td>{op.process_minutes.toFixed(2)} min</td></tr>)}</tbody></table></div>
    <div className="routeTotal"><span>Tool changes {route.summary.tool_changes} · Machine changes {route.summary.machine_changes}</span><strong>{route.summary.total_estimated_minutes.toFixed(2)} estimated min</strong></div>
  </section>
}

export default function App() {
  const [form,setForm] = useState<Request>({part_name:'Research demo bracket',material:'aluminum',features:defaultFeatures,
    available_machines:['cnc_mill','drill_press'], available_tools:allTools})
  const [result,setResult] = useState<Response|null>(null)
  const [step,setStep] = useState<GeometryResponse|null>(null)
  const [apiOnline,setApiOnline] = useState(false)
  const [busy,setBusy] = useState(false)
  const [error,setError] = useState('')
  useEffect(()=>{ api.health().then(()=>setApiOnline(true)).catch(()=>setApiOnline(false)) },[])
  const setField = <K extends keyof Request>(key: K, value:Request[K]) => setForm(v=>({...v,[key]:value}))
  const editFeature = (i:number, patch:Partial<Feature>) => setForm(prev=>({ ...prev,
    features:prev.features.map((f,j)=>i===j?{...f,...patch}:f) }))
  const add = () => setField('features',[...form.features,{id:`feature-${Date.now()}`,kind:'hole',quantity:1,diameter_mm:8,depth_mm:10,length_mm:30,width_mm:15}])
  async function calculate() {
    setBusy(true); setError(''); setResult(null)
    try {setResult(await api.plan(form))}
    catch (e) {setError(e instanceof Error ? e.message : 'Planning failed')}
    finally {setBusy(false)}
  }
  async function upload(file:File) {
    setBusy(true);setError('');setStep(null)
    try {setStep(await api.step(file))}
    catch(e) {setError(e instanceof Error ? e.message:'STEP analysis failed')}
    finally {setBusy(false)}
  }
  const toggleMachine=(m:Machine)=>setField('available_machines',form.available_machines.includes(m)?form.available_machines.filter(x=>x!==m):[...form.available_machines,m])
  const toggleTool=(t:Tool)=>setField('available_tools',form.available_tools.includes(t)?form.available_tools.filter(x=>x!==t):[...form.available_tools,t])
  function download(){if(!result)return; const blob=new Blob([JSON.stringify({input:form,result},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='ps04-routing-result.json';link.click();URL.revokeObjectURL(url)}
  return <div className="app">
    <aside className="sidebar"><div className="brand"><div className="brandIcon"><Factory size={22}/></div><div><strong>FORGE<span>PATH</span></strong><small>Manufacturing intelligence</small></div></div>
      <div className="nav active"><CircuitBoard size={18}/> Process planning</div>
      <div className="nav"><Layers3 size={18}/> CAD geometry audit</div>
      <div className="nav"><Gauge size={18}/> Research baseline</div>
      <div className="sideBottom">PS 04 · v0.1.0<br/>Rules + optimization baseline<br/><span>ML models: not trained yet</span></div>
    </aside>
    <main className="main"><header className="top"><div><div className="eyebrow">PS 04 / RESEARCH PROTOTYPE</div><h1>Manufacturing Process Planner</h1><p>From manufacturing features to a traceable, constraint-aware process route.</p></div><div className={'status '+(apiOnline?'good':'bad')}><span className="dot"/>{apiOnline?'Backend connected':'Backend offline'}</div></header>
      <div className="notice">Scientific honesty: this is a <b>rule-based, sequential routing baseline</b>, not a trained ML model or production-certified CAM planner. Enter verified features manually. All times are illustrative estimates pending industrial calibration.</div>
      <div className="workspace"><div className="left">
        <section className="panel"><div className="panelTitle"><h2><span>01</span> Part configuration</h2><RotateCcw size={17} role="button" className="clickable" onClick={()=>{setForm({part_name:'Research demo bracket',material:'aluminum',features:defaultFeatures,available_machines:['cnc_mill','drill_press'],available_tools:allTools});setResult(null);setError('')}}/></div>
          <div className="twocol"><label>Part name<input value={form.part_name} onChange={e=>setField('part_name',e.target.value)}/></label><label>Material<select value={form.material} onChange={e=>setField('material',e.target.value as Request['material'])}><option value="aluminum">Aluminum</option><option value="steel">Steel</option></select></label></div>
        </section>
        <section className="panel"><div className="panelTitle"><h2><span>02</span> Manufacturing features</h2><button className="mutedButton" onClick={add}><Plus size={16}/> Add</button></div>
        <p className="hint">Specify one supported feature per row. Dimensions are nominal, in millimeters.</p>
        {form.features.map((f,i)=><div className="feature" key={f.id}><div className="featureTop"><span>F{i+1}</span><input aria-label="Feature ID" value={f.id} onChange={e=>editFeature(i,{id:e.target.value})}/><select aria-label="Feature type" value={f.kind} onChange={e=>editFeature(i,{kind:e.target.value as Kind})}><option value="planar_face">Planar face</option><option value="hole">Hole</option><option value="pocket">Pocket</option><option value="slot">Slot</option></select><button className="iconButton" aria-label="Delete feature" onClick={()=>setField('features',form.features.filter((_,j)=>i!==j))}><Trash2 size={15}/></button></div>
        <div className="featureDimensions"><label>Qty<input type="number" min="1" max="100" value={f.quantity} onChange={e=>editFeature(i,{quantity:Number(e.target.value)})}/></label>
        {(f.kind==='hole'?['diameter_mm','depth_mm']:f.kind==='planar_face'?['length_mm','width_mm']:['length_mm','width_mm','depth_mm']).map(k=><label key={k}>{pretty(k)}<input type="number" min="0.1" step="any" value={f[k as keyof Feature] as number} onChange={e=>editFeature(i,{[k]:Number(e.target.value)})}/></label>)}</div></div>)}
        </section>
        <section className="panel"><div className="panelTitle"><h2><span>03</span> Resource constraints</h2></div><div className="subLabel">Available machines</div><div className="checks">{(['cnc_mill','drill_press'] as Machine[]).map(m=><label key={m}><input type="checkbox" checked={form.available_machines.includes(m)} onChange={()=>toggleMachine(m)}/>{pretty(m)}</label>)}</div>
        <div className="subLabel">Available tools</div><div className="checks">{allTools.map(t=><label key={t}><input type="checkbox" checked={form.available_tools.includes(t)} onChange={()=>toggleTool(t)}/>{pretty(t)}</label>)}</div></section>
        <button className="primary" disabled={busy||form.features.length===0} onClick={calculate}>{busy?'Processing…': 'Generate process route'} <ArrowRight size={18}/></button>
        {error&&<div className="error">{error}</div>}
      </div><div className="right">
        <section className="panel"><div className="panelTitle"><h2><span>04</span> Optional STEP geometry summary</h2><FileUp size={18}/></div><p className="hint">Upload a STEP file for shape metadata only. This does not automatically recognize machining features. CadQuery is optional and may need installation.</p><label className="upload">Choose .step or .stp file<input type="file" accept=".step,.stp" onChange={e=>{const f=e.target.files?.[0];if(f)void upload(f);e.target.value=''}}/></label>
        {step&&<div className="cadResults"><b>{step.solid_count} solid(s) · {step.face_count} faces</b><div>Bounding box: {step.bounding_box_mm.x} × {step.bounding_box_mm.y} × {step.bounding_box_mm.z} mm</div><div>Cylindrical faces: {step.cylindrical_face_count} (not confirmed holes)</div><small>{step.classification_note}</small></div>}</section>
        <section className="panel results"><div className="panelTitle"><h2><span>05</span> Generated routing chart</h2>{result&&<button className="mutedButton" onClick={download}><Download size={16}/> Export JSON</button>}</div>
        {!result?<div className="empty"><CircuitBoard size={44}/><strong>No route generated yet</strong><span>Configure the part and press Generate process route to see precedence-aware operations and a baseline comparison.</span></div>:<>
          <div className="stats"><div><span>Baseline estimate</span><strong>{result.baseline.summary.total_estimated_minutes.toFixed(2)}<small> min</small></strong></div><div><span>Optimized estimate</span><strong>{result.optimized.summary.total_estimated_minutes.toFixed(2)}<small> min</small></strong></div><div><span>Illustrative improvement</span><strong>{result.improvement_percent.toFixed(1)}<small>%</small></strong></div></div>
          <div className="flow"><b>Precedence-aware sequence</b><div className="steps">{result.optimized.operations.map((o,i)=><div className="step" key={o.id}><span className="stepNum">{i+1}</span><div><strong>{pretty(o.action)}</strong><small>{o.feature_id} · {pretty(o.machine)} · {o.process_minutes.toFixed(2)} min</small></div></div>)}</div></div>
          <RouteTable title="Optimized process route" route={result.optimized}/>
          <RouteTable title="Feature-order baseline" route={result.baseline}/>
          <div className="hint">{result.scope_note}</div>
        </>}
        </section>
      </div></div><footer>FORGEPATH · Research prototype · Never use estimated process times or tool routes without manufacturing engineer validation.</footer>
    </main></div>
}
