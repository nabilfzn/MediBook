const chat = document.getElementById("chat");
const form = document.getElementById("form");
const inp  = document.getElementById("inp");
const btn  = document.getElementById("btn");

function addMsg(text, me=false){
  const div = document.createElement("div");
  div.className = "msg"+(me?" me":"");
  div.innerHTML = `<span class="bubble">${text.replace(/\n/g,"<br>")}</span>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

addMsg("Halo! Aku bisa bantu rekomendasi buku dari perpustakaan. Ceritakan apa yang ingin kamu baca dan bagaimana perasaanmu.");

form.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const text = inp.value.trim();
  if(!text) return;
  addMsg(text, true);
  inp.value = ""; btn.disabled = true;

  try{
    const res = await fetch("/chat", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    addMsg(data.reply || "(tidak ada balasan)");
  }catch(err){
    addMsg("Maaf, terjadi kesalahan.");
  }finally{
    btn.disabled = false;
  }
});
