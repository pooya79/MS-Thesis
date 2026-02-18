(() => {
  const toggleButton = document.querySelector("[data-sidebar-toggle]");
  const sidebar = document.getElementById("left-sidebar");

  if (toggleButton && sidebar) {
    toggleButton.addEventListener("click", () => {
      const isOpen = sidebar.classList.toggle("open");
      toggleButton.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });
  }

  const tocLinks = Array.from(document.querySelectorAll("[data-toc-link]"));
  if (tocLinks.length === 0) {
    return;
  }

  const sections = tocLinks
    .map((link) => document.querySelector(link.getAttribute("href") || ""))
    .filter((section) => section instanceof HTMLElement);

  if (sections.length === 0) {
    return;
  }

  const setActiveLink = (id) => {
    tocLinks.forEach((link) => {
      const isActive = link.getAttribute("href") === `#${id}`;
      link.classList.toggle("active", isActive);
    });
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visibleEntry = entries.find((entry) => entry.isIntersecting);
      if (visibleEntry?.target?.id) {
        setActiveLink(visibleEntry.target.id);
      }
    },
    {
      rootMargin: "-30% 0px -55% 0px",
      threshold: [0, 0.2, 0.6],
    },
  );

  sections.forEach((section) => observer.observe(section));
})();
