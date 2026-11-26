
const toggle=document.getElementById('mobileToggle'),nav=document.getElementById('mainNav');
if(toggle){toggle.addEventListener('click',()=>{const open=nav.classList.toggle('open');toggle.setAttribute('aria-expanded',open)})}
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',e=>{const el=document.querySelector(a.getAttribute('href'));if(el){e.preventDefault();el.scrollIntoView({behavior:'smooth'});nav?.classList.remove('open')}}));
document.getElementById('year').textContent=new Date().getFullYear();
