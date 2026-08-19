/**
 * ΟΚΤΩ · Σχολή Χορού — η μοναδική JavaScript του site.
 *
 * ΑΡΧΗ: η σελίδα δουλεύει και χωρίς αυτό το αρχείο. Ό,τι υπάρχει εδώ είναι
 * βελτίωση, όχι προϋπόθεση — αν αποτύχει, το περιεχόμενο παραμένει ορατό.
 *
 * Περιεχόμενα:
 *   1. Μενού σε κινητό
 *   2. «Σήμερα στο ΟΚΤΩ» — επιλογή της τρέχουσας μέρας στην αρχική
 *   3. Φίλτρα στο εβδομαδιαίο πρόγραμμα
 *   4. Μεγέθυνση φωτογραφιών (lightbox)
 */

/* =====================================================================
   1. ΜΕΝΟΥ ΣΕ ΚΙΝΗΤΟ
   ================================================================== */

function setupNav() {
  const toggle = document.querySelector(".nav-toggle");
  const nav = document.querySelector(".site-nav");
  if (!toggle || !nav) return;

  toggle.addEventListener("click", () => {
    const isOpen = nav.classList.toggle("is-open");
    // Το aria-expanded ενημερώνει τους αναγνώστες οθόνης — και το CSS,
    // που πάνω του βασίζει το «Χ» του εικονιδίου.
    toggle.setAttribute("aria-expanded", String(isOpen));
  });

  // Κλείσιμο με Escape, όπως περιμένει κανείς από κάθε ανοιχτό πάνελ.
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && nav.classList.contains("is-open")) {
      nav.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
      toggle.focus();
    }
  });
}

/* =====================================================================
   2. «ΣΗΜΕΡΑ ΣΤΟ ΟΚΤΩ»
   ---------------------------------------------------------------------
   Στην αρχική οι μέρες είναι καρτέλες φτιαγμένες με radio buttons: δουλεύουν
   με το ποντίκι και με το πληκτρολόγιο ακόμη κι αν δεν τρέξει καθόλου
   JavaScript. Εδώ κάνουμε ένα μόνο πράγμα — προεπιλέγουμε τη σημερινή μέρα.
   ================================================================== */

function setupToday() {
  const panel = document.querySelector("[data-today]");
  if (!panel) return;

  // Η getDay() επιστρέφει 0 για Κυριακή. Την Κυριακή η σχολή είναι κλειστή,
  // οπότε δείχνουμε τη Δευτέρα.
  const keys = ["mon", "mon", "tue", "wed", "thu", "fri", "sat"];
  const today = keys[new Date().getDay()];

  const radio = panel.querySelector(`input[value="${today}"]`);
  if (radio) radio.checked = true;

  // Σήμανση «σήμερα» δίπλα στο όνομα της μέρας.
  const label = panel.querySelector(`[data-day-label="${today}"]`);
  if (label) label.dataset.isToday = "true";
}

/* =====================================================================
   3. ΦΙΛΤΡΑ ΠΡΟΓΡΑΜΜΑΤΟΣ
   ---------------------------------------------------------------------
   Δύο ομάδες κουμπιών (είδος χορού / επίπεδο). Κάθε ώρα του προγράμματος
   φέρει data-class και data-level· κρύβουμε όσες δεν ταιριάζουν και στα δύο.
   ================================================================== */

function setupScheduleFilters() {
  const board = document.querySelector("[data-schedule]");
  if (!board) return;

  const chips = board.querySelectorAll(".chip");
  const sessions = board.querySelectorAll("[data-class]");

  // Η τρέχουσα επιλογή ανά ομάδα· "all" σημαίνει «χωρίς φίλτρο».
  const selected = { class: "all", level: "all" };

  function apply() {
    sessions.forEach((item) => {
      const matches =
        (selected.class === "all" || item.dataset.class === selected.class) &&
        (selected.level === "all" || item.dataset.level === selected.level);
      item.hidden = !matches;
    });

    // Αν μια μέρα έμεινε χωρίς ώρες, το λέμε αντί να αφήσουμε κενό.
    board.querySelectorAll("[data-day]").forEach((day) => {
      const visible = day.querySelectorAll("[data-class]:not([hidden])").length;
      const empty = day.querySelector(".day__empty");
      if (empty) empty.hidden = visible > 0;
    });
  }

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const group = chip.dataset.group; // "class" ή "level"
      selected[group] = chip.dataset.value;

      // Μόνο ένα κουμπί ανά ομάδα μένει πατημένο.
      board
        .querySelectorAll(`.chip[data-group="${group}"]`)
        .forEach((other) => other.setAttribute("aria-pressed", String(other === chip)));

      apply();
    });
  });

  apply();
}

/* =====================================================================
   4. LIGHTBOX ΓΚΑΛΕΡΙ
   ---------------------------------------------------------------------
   Χρησιμοποιεί το στοιχείο <dialog>: ο browser αναλαμβάνει το «σκοτείνιασμα»
   του φόντου, το κλείσιμο με Escape και την επιστροφή της εστίασης.
   ================================================================== */

function setupLightbox() {
  const dialog = document.querySelector(".lightbox");
  const buttons = Array.from(document.querySelectorAll("[data-lightbox]"));
  if (!dialog || buttons.length === 0 || typeof dialog.showModal !== "function") return;

  const image = dialog.querySelector(".lightbox__image");
  const caption = dialog.querySelector(".lightbox__caption");
  const closeButton = dialog.querySelector(".lightbox__close");
  let current = 0;

  function show(index) {
    // Κυκλική μετακίνηση: μετά την τελευταία έρχεται η πρώτη.
    current = (index + buttons.length) % buttons.length;
    const button = buttons[current];
    image.src = button.dataset.lightbox;
    image.alt = button.dataset.alt || "";
    caption.textContent = button.dataset.caption || "";
  }

  buttons.forEach((button, index) => {
    button.addEventListener("click", () => {
      show(index);
      dialog.showModal();
    });
  });

  closeButton?.addEventListener("click", () => dialog.close());

  // Κλικ έξω από την εικόνα κλείνει το παράθυρο.
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  // Βέλη για επόμενη/προηγούμενη φωτογραφία.
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") show(current + 1);
    if (event.key === "ArrowLeft") show(current - 1);
  });
}

/* =====================================================================
   Εκκίνηση
   ================================================================== */

setupNav();
setupToday();
setupScheduleFilters();
setupLightbox();
