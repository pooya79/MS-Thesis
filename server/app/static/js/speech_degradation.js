(() => {
  const tabButtons = Array.from(document.querySelectorAll("[data-tab-button]"));
  const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));

  const activateTab = (name) => {
    tabButtons.forEach((button) => {
      const isActive = button.dataset.tabButton === name;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.dataset.tabPanel === name);
    });
  };

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tabButton));
  });

  const gainInput = document.querySelector("[data-gain-input]");
  const gainValue = document.querySelector("[data-gain-value]");
  if (gainInput && gainValue) {
    gainInput.addEventListener("input", () => {
      gainValue.textContent = gainInput.value;
    });
  }

  const channelSelect = document.querySelector("[data-channel-select]");
  const codecSelect = document.querySelector("[data-codec-select]");

  const updateCodecOptions = () => {
    if (!channelSelect || !codecSelect) {
      return;
    }
    const channel = channelSelect.value;
    let selectedStillValid = false;
    Array.from(codecSelect.options).forEach((option) => {
      const allowed = option.dataset.channel === "both" || option.dataset.channel === channel;
      option.disabled = !allowed;
      if (option.selected && allowed) {
        selectedStillValid = true;
      }
    });
    if (!selectedStillValid) {
      const fallback = Array.from(codecSelect.options).find((option) => !option.disabled);
      if (fallback) {
        fallback.selected = true;
      }
    }
  };

  if (channelSelect && codecSelect) {
    channelSelect.addEventListener("change", updateCodecOptions);
    updateCodecOptions();
  }

  if (document.querySelector(".audio-result")) {
    activateTab("results");
  }
})();
