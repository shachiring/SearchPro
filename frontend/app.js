const apiUrlInput = document.getElementById('apiUrl')
const fileInput = document.getElementById('fileInput')
const urlInput = document.getElementById('urlInput')
const minScore = document.getElementById('minScore')
const minVal = document.getElementById('minVal')
const topN = document.getElementById('topN')
const searchBtn = document.getElementById('searchBtn')
const resultsEl = document.getElementById('results')
const queryPreview = document.getElementById('queryPreview')

minScore.addEventListener('input', () => minVal.textContent = minScore.value)

function getApiBase(){
  const v = apiUrlInput.value.trim()
  return v || window.location.origin
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

  searchBtn.disabled = true
  searchBtn.textContent = 'Searching...'
  try{
    const res = await fetch(apiBase + '/search', { method: 'POST', body: fd })
    const data = await res.json()
    if (data.error){
      resultsEl.textContent = data.error
    } else {
      const results = data.results || []
      if (!results.length) resultsEl.textContent = 'No results'
      results.forEach(r => {
        const card = document.createElement('div')
        card.className = 'card'
        const img = document.createElement('img')
        // use absolute URL if apiBase provided
        const imgUrl = r.image_url.startsWith('/') ? apiBase + r.image_url : r.image_url
        img.src = imgUrl
        card.appendChild(img)
        const meta = document.createElement('div')
        meta.className = 'meta'
        meta.innerHTML = `<strong>${r.name}</strong><br/>${r.category} — $${r.price}<br/>Similarity: ${r.score.toFixed(3)}`
        card.appendChild(meta)
        resultsEl.appendChild(card)
      })
    }
  }catch(e){
    resultsEl.textContent = 'Request failed: '+e.message
  }finally{
    searchBtn.disabled = false
    searchBtn.textContent = 'Search'
  }
})
