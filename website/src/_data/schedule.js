/**
 * Εβδομαδιαίο πρόγραμμα μαθημάτων.
 *
 * ⚠️  ΕΙΚΟΝΙΚΑ ΔΕΔΟΜΕΝΑ (demo).
 *
 * ΔΟΜΗ: μία και μόνη «επίπεδη» λίστα ωρών (`sessions`). Από αυτήν παράγονται
 * και ο εβδομαδιαίος πίνακας (/programma/) και οι ώρες που εμφανίζονται μέσα
 * σε κάθε σελίδα μαθήματος. Έτσι, όταν αλλάξει το πρόγραμμα, αλλάζει ΕΝΑ αρχείο.
 *
 * Κάθε ώρα συνδέεται με τα υπόλοιπα δεδομένα μέσω slugs:
 *   class   → το όνομα αρχείου στο src/mathimata/  (π.χ. "latin" → latin.md)
 *   teacher → το όνομα αρχείου στο src/kathigites/ (π.χ. "marina-kostea")
 *   level   → κλειδί από το `levels` παρακάτω
 */

export default {
  /* Οι μέρες με τη σειρά που εμφανίζονται στον πίνακα.
     `short` = συντομογραφία για τις στήλες σε μικρές οθόνες. */
  days: [
    { key: "mon", label: "Δευτέρα", short: "Δευ" },
    { key: "tue", label: "Τρίτη", short: "Τρί" },
    { key: "wed", label: "Τετάρτη", short: "Τετ" },
    { key: "thu", label: "Πέμπτη", short: "Πέμ" },
    { key: "fri", label: "Παρασκευή", short: "Παρ" },
    { key: "sat", label: "Σάββατο", short: "Σάβ" },
  ],

  /* Επίπεδα. Το κλειδί μπαίνει στο HTML (data-level) για το φιλτράρισμα,
     η ετικέτα εμφανίζεται στον χρήστη. */
  levels: {
    arxarioi: "Αρχάριοι",
    meso: "Μέσο επίπεδο",
    proxorimenoi: "Προχωρημένοι",
    ola: "Όλα τα επίπεδα",
    paidiko: "Παιδικό",
  },

  /*
   * Οι ώρες της εβδομάδας.
   * Οι αίθουσες είναι δύο: «Α» (μεγάλη, με μπάρες) και «Β».
   * Καμία αίθουσα και κανένας καθηγητής δεν εμφανίζεται σε δύο μαθήματα
   * την ίδια ώρα — αν προσθέσεις ώρες, κράτα τον ίδιο κανόνα.
   */
  sessions: [
    /* --- Δευτέρα --- */
    { day: "mon", start: "17:00", end: "18:00", class: "paidiko", teacher: "dafni-antoniou", level: "paidiko", room: "Β" },
    { day: "mon", start: "18:00", end: "19:00", class: "baleto", teacher: "dafni-antoniou", level: "paidiko", room: "Α", note: "7–10 ετών" },
    { day: "mon", start: "19:00", end: "20:15", class: "baleto", teacher: "eleni-papadaki", level: "arxarioi", room: "Α" },
    { day: "mon", start: "19:00", end: "20:15", class: "hip-hop", teacher: "nikos-vlachakis", level: "meso", room: "Β" },
    { day: "mon", start: "20:15", end: "21:30", class: "sygxronos", teacher: "eleni-papadaki", level: "meso", room: "Α" },
    { day: "mon", start: "20:30", end: "21:45", class: "latin", teacher: "marina-kostea", level: "arxarioi", room: "Β" },

    /* --- Τρίτη --- */
    { day: "tue", start: "17:30", end: "18:30", class: "paidiko", teacher: "dafni-antoniou", level: "paidiko", room: "Β" },
    { day: "tue", start: "18:30", end: "19:45", class: "hip-hop", teacher: "nikos-vlachakis", level: "arxarioi", room: "Β" },
    { day: "tue", start: "19:00", end: "20:15", class: "sygxronos", teacher: "zoi-markaki", level: "arxarioi", room: "Α" },
    { day: "tue", start: "20:15", end: "21:30", class: "baleto", teacher: "eleni-papadaki", level: "meso", room: "Α" },
    { day: "tue", start: "20:30", end: "21:45", class: "oriental", teacher: "zoi-markaki", level: "ola", room: "Β" },
    { day: "tue", start: "21:30", end: "22:45", class: "tango", teacher: "marina-kostea", level: "ola", room: "Α" },

    /* --- Τετάρτη --- */
    { day: "wed", start: "17:00", end: "18:00", class: "paidiko", teacher: "dafni-antoniou", level: "paidiko", room: "Β" },
    { day: "wed", start: "18:00", end: "19:00", class: "baleto", teacher: "dafni-antoniou", level: "paidiko", room: "Α", note: "7–10 ετών" },
    { day: "wed", start: "19:00", end: "20:15", class: "sygxronos", teacher: "eleni-papadaki", level: "proxorimenoi", room: "Α" },
    { day: "wed", start: "19:00", end: "20:15", class: "latin", teacher: "marina-kostea", level: "meso", room: "Β" },
    { day: "wed", start: "20:15", end: "21:30", class: "baleto", teacher: "eleni-papadaki", level: "proxorimenoi", room: "Α" },
    { day: "wed", start: "20:30", end: "21:45", class: "hip-hop", teacher: "nikos-vlachakis", level: "ola", room: "Β" },

    /* --- Πέμπτη --- */
    { day: "thu", start: "17:30", end: "18:30", class: "paidiko", teacher: "dafni-antoniou", level: "paidiko", room: "Β" },
    { day: "thu", start: "18:30", end: "19:45", class: "baleto", teacher: "dafni-antoniou", level: "paidiko", room: "Α", note: "11–14 ετών" },
    { day: "thu", start: "19:00", end: "20:15", class: "hip-hop", teacher: "nikos-vlachakis", level: "meso", room: "Β" },
    { day: "thu", start: "20:00", end: "21:15", class: "paradosiakoi", teacher: "thanasis-rigas", level: "arxarioi", room: "Α" },
    { day: "thu", start: "20:30", end: "21:45", class: "sygxronos", teacher: "zoi-markaki", level: "meso", room: "Β" },
    { day: "thu", start: "21:15", end: "22:30", class: "paradosiakoi", teacher: "thanasis-rigas", level: "proxorimenoi", room: "Α" },

    /* --- Παρασκευή --- */
    { day: "fri", start: "18:00", end: "19:00", class: "paidiko", teacher: "dafni-antoniou", level: "paidiko", room: "Β" },
    { day: "fri", start: "19:00", end: "20:15", class: "latin", teacher: "marina-kostea", level: "ola", room: "Α" },
    { day: "fri", start: "19:00", end: "20:15", class: "oriental", teacher: "zoi-markaki", level: "arxarioi", room: "Β" },
    { day: "fri", start: "20:15", end: "21:30", class: "tango", teacher: "marina-kostea", level: "arxarioi", room: "Α" },
    { day: "fri", start: "20:30", end: "21:45", class: "hip-hop", teacher: "nikos-vlachakis", level: "proxorimenoi", room: "Β" },
    { day: "fri", start: "21:30", end: "23:00", class: "tango", teacher: "marina-kostea", level: "ola", room: "Α", note: "Πρακτική – μιλόνγκα" },

    /* --- Σάββατο --- */
    { day: "sat", start: "10:00", end: "11:00", class: "paidiko", teacher: "dafni-antoniou", level: "paidiko", room: "Α" },
    { day: "sat", start: "10:00", end: "11:15", class: "latin", teacher: "marina-kostea", level: "arxarioi", room: "Β" },
    { day: "sat", start: "11:00", end: "12:15", class: "baleto", teacher: "eleni-papadaki", level: "ola", room: "Α" },
    { day: "sat", start: "11:30", end: "12:45", class: "hip-hop", teacher: "nikos-vlachakis", level: "paidiko", room: "Β", note: "Hip Hop Kids, 8–12 ετών" },
    { day: "sat", start: "12:30", end: "13:45", class: "sygxronos", teacher: "eleni-papadaki", level: "ola", room: "Α" },
    { day: "sat", start: "13:00", end: "14:15", class: "paradosiakoi", teacher: "thanasis-rigas", level: "ola", room: "Β" },
  ],
};
