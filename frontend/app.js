const apiUrlInput = document.getElementById('apiUrl')
const fileInput = document.getElementById('fileInput')
const urlInput = document.getElementById('urlInput')
const minScore = document.getElementById('minScore')
const minVal = document.getElementById('minVal')
const topN = document.getElementById('topN')
const searchBtn = document.getElementById('searchBtn')
const resultsEl = document.getElementById('results')
const queryPreview = document.getElementById('queryPreview')

minScore.addEventListener('input', () => minVal.textContent = parseFloat(minScore.value).toFixed(2))

function getApiBase(){
  const v = apiUrlInput.value.trim()
  return v || window.location.origin
}

function setLoading(loading){
  if (loading){
    searchBtn.disabled = true
    searchBtn.textContent = 'Searching...'
    searchBtn.classList.add('loading')
  } else {
    searchBtn.disabled = false
    searchBtn.textContent = 'Search'
    searchBtn.classList.remove('loading')
  }
}

function emptyState(msg){
  resultsEl.innerHTML = `<div class="empty">${msg}</div>`
}

searchBtn.addEventListener('click', async () => {
  resultsEl.innerHTML = ''
  queryPreview.innerHTML = ''
  const apiBase = getApiBase()
  const fd = new FormData()
  if (fileInput.files.length){
    fd.append('file', fileInput.files[0])
  } else if (urlInput.value.trim()){
    fd.append('url', urlInput.value.trim())
  } else {
    alert('Please provide an image file or URL')
    return
  }
  fd.append('min_score', minScore.value)
  fd.append('top_n', topN.value)
  if (fileInput.files.length){
    const reader = new FileReader()
    reader.onload = () => {
      const img = document.createElement('img')
      img.src = reader.result
      queryPreview.appendChild(img)
    }
    reader.readAsDataURL(fileInput.files[0])
  }

  setLoading(true)
  emptyState('Searching — please wait...')
  try{
    const res = await fetch(apiBase + '/search', { method: 'POST', body: fd })
    const data = await res.json()
    if (data.error){
      emptyState(data.error)
    } else {
      const results = data.results || []
      if (!results.length) emptyState('No results found — try lowering similarity')
      else {
        resultsEl.innerHTML = ''
        results.forEach(r => {
          const card = document.createElement('div')
          card.className = 'card'
          const img = document.createElement('img')
          const imgUrl = r.image_url.startsWith('/') ? apiBase + r.image_url : r.image_url
          img.src = imgUrl
          card.appendChild(img)
          const meta = document.createElement('div')
          meta.className = 'meta'
          meta.innerHTML = `<strong>${r.name}</strong><div class="meta-row">${r.category} • $${r.price} <span class="badge">${r.score.toFixed(3)}</span></div>`
          card.appendChild(meta)
          resultsEl.appendChild(card)
        })
      }
    }
  }catch(e){
    emptyState('Request failed: '+e.message)
  }finally{
    setLoading(false)
  }
})
