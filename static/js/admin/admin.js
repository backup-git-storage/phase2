(() => {
  const sidebar = document.getElementById("sidebar");
  const menu = document.getElementById("mobileMenu");
  const backdrop = document.getElementById("sidebarBackdrop");
  const navLinks = Array.from(document.querySelectorAll(".side-nav a"));

  function setSidebarOpen(open) {
    if (sidebar) sidebar.classList.toggle("open", open);
    if (backdrop) backdrop.classList.toggle("open", open);
    if (menu) menu.setAttribute("aria-expanded", String(open));
  }

  if (menu) {
    menu.addEventListener("click", () => {
      setSidebarOpen(!sidebar?.classList.contains("open"));
    });
  }

  if (backdrop) {
    backdrop.addEventListener("click", () => setSidebarOpen(false));
  }

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.forEach((item) => {
        item.classList.remove("active");
        item.removeAttribute("aria-current");
      });
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
      setSidebarOpen(false);
    });
  });

  const sections = navLinks
    .map((link) => {
      const href = link.getAttribute("href");
      return href?.startsWith("#") ? document.querySelector(href) : null;
    })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const matchingLink = navLinks.find(
          (link) => link.getAttribute("href") === `#${visible.target.id}`
        );
        if (!matchingLink) return;
        navLinks.forEach((link) => {
          link.classList.remove("active");
          link.removeAttribute("aria-current");
        });
        matchingLink.classList.add("active");
        matchingLink.setAttribute("aria-current", "page");
      },
      { rootMargin: "-20% 0px -65%", threshold: [0.1, 0.35] }
    );
    sections.forEach((section) => observer.observe(section));
  }

  const search = document.getElementById("userSearch");
  const roleFilter = document.getElementById("roleFilter");
  const statusFilter = document.getElementById("statusFilter");
  const tableBody = document.getElementById("userTableBody");
  const tableWrap = document.getElementById("userTableWrap");
  const tableEmpty = document.getElementById("userTableEmpty");
  const filteredEmpty = document.getElementById("filteredEmpty");
  const countBadge = document.getElementById("userCountBadge");
  const activityEmpty = document.getElementById("activityEmpty");
  const activityList = document.getElementById("activityList");

  const rows = tableBody
    ? Array.from(tableBody.querySelectorAll("tr"))
    : [];

  if (tableWrap) tableWrap.hidden = rows.length === 0;
  if (tableEmpty) tableEmpty.hidden = rows.length > 0;
  if (countBadge) {
    countBadge.textContent = rows.length
      ? `${rows.length} account${rows.length === 1 ? "" : "s"}`
      : "No records loaded";
  }

  function filterUsers() {
    if (!rows.length) return;
    const term = (search?.value || "").trim().toLowerCase();
    const role = roleFilter?.value || "";
    const status = statusFilter?.value || "";
    let visibleCount = 0;

    rows.forEach((row) => {
      const searchable = [
        row.dataset.name,
        row.dataset.hospital,
        row.dataset.email,
      ].join(" ");
      const matchesSearch = !term || searchable.includes(term);
      const matchesRole = !role || row.dataset.role === role;
      const matchesStatus = !status || row.dataset.status === status;
      const visible = matchesSearch && matchesRole && matchesStatus;
      row.hidden = !visible;
      if (visible) visibleCount += 1;
    });

    if (countBadge) {
      countBadge.textContent = visibleCount === rows.length
        ? `${rows.length} account${rows.length === 1 ? "" : "s"}`
        : `${visibleCount} of ${rows.length} accounts`;
    }
    if (filteredEmpty) filteredEmpty.hidden = visibleCount !== 0;
  }

  if (search) search.addEventListener("input", filterUsers);
  if (roleFilter) roleFilter.addEventListener("change", filterUsers);
  if (statusFilter) statusFilter.addEventListener("change", filterUsers);

  if (activityList) {
    const activityEvents = activityList.querySelectorAll(".activity-event").length;
    if (activityEvents > 0) {
      activityList.hidden = false;
      if (activityEmpty) activityEmpty.hidden = true;
    }
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setSidebarOpen(false);
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 1050) setSidebarOpen(false);
  });
})();
