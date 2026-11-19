const featureDefs = [
 ["battery_power","Battery power (mAh)",500,2000,1],["blue","Bluetooth",0,1,1],
 ["clock_speed","Clock speed (GHz)",0.5,3,0.1],["dual_sim","Dual SIM",0,1,1],
 ["fc","Front camera (MP)",0,19,1],["four_g","4G support",0,1,1],
 ["int_memory","Internal memory (GB)",2,64,1],["m_dep","Mobile depth (cm)",0.1,1,0.1],
 ["mobile_wt","Mobile weight (g)",80,200,1],["n_cores","Number of cores",1,8,1],
 ["pc","Primary camera (MP)",0,20,1],["px_height","Pixel resolution height",0,1960,1],
 ["px_width","Pixel resolution width",500,2000,1],["ram","RAM (MB)",256,4000,1],
 ["sc_h","Screen height (cm)",5,19,1],["sc_w","Screen width (cm)",0,18,1],
 ["talk_time","Talk time (hours)",2,20,1],["three_g","3G support",0,1,1],
 ["touch_screen","Touch screen",0,1,1],["wifi","Wi-Fi",0,1,1]
];
const defaults = {battery_power:1200,blue:1,clock_speed:2.0,dual_sim:1,fc:5,four_g:1,int_memory:32,m_dep:0.5,mobile_wt:150,n_cores:4,pc:12,px_height:900,px_width:1600,ram:2000,sc_h:12,sc_w:7,talk_time:12,three_g:1,touch_screen:1,wifi:1};
const classNames = ["Low Cost","Medium Cost","High Cost","Very High Cost"];
const formEl = document.getElementById("feature-form");
formEl.innerHTML = featureDefs.map(([key,label,min,max,step]) => `<div class="feature-field"><label for="${key}">${label}</label><input id="${key}" name="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${defaults[key]}" required></div>`).join("");
function showPage(name){
 document.querySelectorAll(".page").forEach(p=>p.classList.toggle("active",p.id===`page-${name}`));
 document.querySelectorAll(".nav-link").forEach(b=>b.classList.toggle("active",b.dataset.page===name));
 document.getElementById("page-title").textContent={dashboard:"Dashboard",predict:"Price Prediction",comparison:"Model Comparison",visuals:"Visualizations"}[name];
}
document.querySelectorAll(".nav-link").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.page)));
document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.go)));
function renderTable(models){
 if(!models?.length)return '<div class="empty-state">Train the models to view measured performance.</div>';
 return `<table class="data-table"><thead><tr><th>MODEL</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1-SCORE</th></tr></thead><tbody>${models.map(m=>`<tr><td>${m.model}</td><td class="score">${(m.accuracy*100).toFixed(2)}%</td><td>${(m.precision*100).toFixed(2)}%</td><td>${(m.recall*100).toFixed(2)}%</td><td>${(m.f1_score*100).toFixed(2)}%</td></tr>`).join("")}</tbody></table>`;
}
async function loadMetrics(){
 try{
  const res=await fetch("/api/metrics");const data=await res.json();
  if(data.available){document.getElementById("dashboard-metrics").innerHTML=renderTable(data.models);document.getElementById("comparison-table").innerHTML=renderTable(data.models);}
 }catch(e){console.warn("Metrics unavailable",e);}
}
async function loadHealth(){
 try{const r=await fetch("/api/health");const h=await r.json();const ready=h.models_ready;
  document.getElementById("api-status").textContent=ready?"API · Models ready":"API · Models not trained";
  document.getElementById("side-status").textContent=ready?"Three models loaded":"Add dataset and train models";
  document.querySelector(".pulse").style.background=ready?"#4bd6a3":"#e9a94e";
 }catch(e){document.getElementById("api-status").textContent="API unavailable";document.getElementById("side-status").textContent="Backend unavailable";}
}
document.getElementById("prediction-form").addEventListener("submit",async e=>{
 e.preventDefault();const btn=document.getElementById("predict-btn");const out=document.getElementById("prediction-results");
 const mobile={};featureDefs.forEach(([key])=>mobile[key]=Number(document.getElementById(key).value));
 btn.disabled=true;btn.textContent="Predicting…";out.innerHTML='<div class="empty-state">Running model inference…</div>';
 try{
  const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mobile,model:document.getElementById("model-choice").value})});
  const data=await response.json();if(!response.ok)throw new Error(data.detail||"Prediction request failed.");
  out.innerHTML=data.results.map(result=>`<article class="result-card"><p class="eyebrow">${result.model.toUpperCase()}</p><h4>Predicted category</h4><div class="result-big">${result.predicted_category}</div><div class="result-meta">Class ${result.predicted_class} · Confidence ${(result.confidence*100).toFixed(2)}%</div><b style="font-size:11px">Class probability distribution</b>${[0,1,2,3].map(i=>`<div class="prob-row"><span>${classNames[i]}</span><div class="prob-track"><div class="prob-fill" style="width:${(result.probabilities[String(i)]*100).toFixed(2)}%"></div></div><span>${(result.probabilities[String(i)]*100).toFixed(1)}%</span></div>`).join("")}</article>`).join("");
 }catch(err){out.innerHTML=`<div class="result-card"><b>Unable to predict</b><p class="result-meta">${err.message}</p></div>`;}
 finally{btn.disabled=false;btn.innerHTML='Predict price range <span>→</span>';}
});
document.getElementById("prediction-form").addEventListener("reset",()=>setTimeout(()=>document.getElementById("prediction-results").innerHTML="",0));
loadHealth();loadMetrics();
