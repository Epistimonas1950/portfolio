/**
 * Ρυθμός — Σχολή Χορού
 * Κεντρικό αρχείο ρυθμίσεων του Eleventy (v3, ES modules).
 *
 * Δομή του project:
 *   src/            → όλο το περιεχόμενο (σελίδες, δεδομένα, assets)
 *   _site/          → το αποτέλεσμα του build (δημιουργείται αυτόματα)
 *
 * Τεκμηρίωση: https://www.11ty.dev/docs/config/
 */

import eleventyNavigationPlugin from "@11ty/eleventy-navigation";

export default function (eleventyConfig) {
  /* ---------------------------------------------------------------------
   * 1. Plugins
   * ------------------------------------------------------------------- */

  // Φτιάχνει αυτόματα το μενού πλοήγησης από το `eleventyNavigation`
  // που δηλώνουμε στο front matter κάθε σελίδας.
  eleventyConfig.addPlugin(eleventyNavigationPlugin);

  /* ---------------------------------------------------------------------
   * 2. Passthrough copy — αρχεία που αντιγράφονται αυτούσια
   * ------------------------------------------------------------------- */

  // CSS, JS, εικόνες και favicon δεν περνάνε από template engine·
  // απλώς αντιγράφονται στο _site/ διατηρώντας τη διαδρομή τους.
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

  // Κάνε reload τον browser όταν αλλάζει κάτι μέσα στο assets/.
  eleventyConfig.addWatchTarget("src/assets/");

  /* ---------------------------------------------------------------------
   * 3. Φίλτρα ημερομηνίας (ελληνική μορφοποίηση)
   * ------------------------------------------------------------------- */

  /**
   * Ημερομηνία σε αναγνώσιμη ελληνική μορφή: «12 Σεπτεμβρίου 2025».
   * Χρησιμοποιεί το ενσωματωμένο Intl του Node — δεν χρειάζεται βιβλιοθήκη.
   */
  eleventyConfig.addFilter("dateGR", (value) => {
    return new Intl.DateTimeFormat("el-GR", {
      day: "numeric",
      month: "long",
      year: "numeric",
      timeZone: "Europe/Athens",
    }).format(new Date(value));
  });

  /** Ημερομηνία σε μορφή YYYY-MM-DD, για το <time datetime="…">. */
  eleventyConfig.addFilter("isoDate", (value) => {
    return new Date(value).toISOString().slice(0, 10);
  });

  /* ---------------------------------------------------------------------
   * 4. Βοηθητικά φίλτρα για συσχέτιση δεδομένων
   *
   * Τα μαθήματα, οι καθηγητές και το πρόγραμμα συνδέονται μεταξύ τους με
   * slugs (π.χ. "sygxronos"). Αυτά τα φίλτρα κρατούν τα templates καθαρά,
   * χωρίς loops μέσα σε loops.
   * ------------------------------------------------------------------- */

  /**
   * Διαβάζει τιμή από αντικείμενο με «διαδρομή» κλειδιών.
   * get(item, "data.slug") → item.data.slug
   * Χρειάζεται επειδή στα collections του Eleventy το front matter
   * βρίσκεται μέσα στο `.data`.
   */
  const get = (object, path) =>
    path.split(".").reduce((value, key) => (value ?? {})[key], object);

  /** Επιστρέφει ΟΛΑ τα στοιχεία ενός πίνακα όπου item[key] === value. */
  eleventyConfig.addFilter("filterBy", (array = [], key, value) =>
    array.filter((item) => get(item, key) === value)
  );

  /** Επιστρέφει το ΠΡΩΤΟ στοιχείο ενός πίνακα όπου item[key] === value. */
  eleventyConfig.addFilter("findBy", (array = [], key, value) =>
    array.find((item) => get(item, key) === value)
  );

  /** Κρατάει τα πρώτα Ν στοιχεία (π.χ. τα 3 τελευταία νέα στην αρχική). */
  eleventyConfig.addFilter("limit", (array = [], n) => array.slice(0, n));

  /**
   * Κρατάει ένα μόνο στοιχείο για κάθε διαφορετική τιμή του `key`.
   * Χρήσιμο π.χ. για να δείξουμε ποια μαθήματα διδάσκει ένας καθηγητής:
   * το πρόγραμμα τον έχει σε 6 ώρες, αλλά μόνο σε 2 διαφορετικά μαθήματα.
   */
  eleventyConfig.addFilter("uniqueBy", (array = [], key) => {
    const seen = new Set();
    return array.filter((item) => {
      const value = get(item, key);
      if (seen.has(value)) return false;
      seen.add(value);
      return true;
    });
  });

  /* ---------------------------------------------------------------------
   * 5. Collections
   *
   * Το Eleventy φτιάχνει αυτόματα collections από τα tags. Ορίζουμε ρητά
   * μόνο ό,τι χρειάζεται συγκεκριμένη σειρά ταξινόμησης.
   * ------------------------------------------------------------------- */

  // Μαθήματα: με τη σειρά που ορίζει το πεδίο `order` στο front matter.
  eleventyConfig.addCollection("classes", (collectionApi) => {
    return collectionApi
      .getFilteredByTag("class")
      .sort((a, b) => (a.data.order ?? 99) - (b.data.order ?? 99));
  });

  // Καθηγητές: επίσης με βάση το `order`.
  eleventyConfig.addCollection("teachers", (collectionApi) => {
    return collectionApi
      .getFilteredByTag("teacher")
      .sort((a, b) => (a.data.order ?? 99) - (b.data.order ?? 99));
  });

  // Νέα / ανακοινώσεις: νεότερα πρώτα.
  eleventyConfig.addCollection("news", (collectionApi) => {
    return collectionApi
      .getFilteredByTag("post")
      .sort((a, b) => b.date - a.date);
  });

  /*
   * Ευρετήρια slug → σελίδα.
   * Το εβδομαδιαίο πρόγραμμα (_data/schedule.js) αναφέρεται σε μαθήματα και
   * καθηγητές μόνο με το slug τους. Τα ευρετήρια επιτρέπουν στα templates να
   * βρίσκουν αμέσως τον τίτλο ή το χρώμα ενός μαθήματος:
   *   collections.classBySlug["latin"].data.title
   */
  const indexBySlug = (pages) =>
    Object.fromEntries(pages.map((page) => [page.data.slug, page]));

  eleventyConfig.addCollection("classBySlug", (collectionApi) =>
    indexBySlug(collectionApi.getFilteredByTag("class"))
  );

  eleventyConfig.addCollection("teacherBySlug", (collectionApi) =>
    indexBySlug(collectionApi.getFilteredByTag("teacher"))
  );

  /* ---------------------------------------------------------------------
   * 6. Shortcodes
   * ------------------------------------------------------------------- */

  // Τρέχον έτος — για το copyright στο footer.
  eleventyConfig.addShortcode("year", () => `${new Date().getFullYear()}`);

  /* ---------------------------------------------------------------------
   * 7. Ρυθμίσεις dev server
   * ------------------------------------------------------------------- */

  eleventyConfig.setServerOptions({
    port: 8080,
    showAllHosts: false,
  });
}

/**
 * Ρυθμίσεις φακέλων.
 * Τα layouts μένουν μέσα στο _includes/layouts/ (η προεπιλογή του Eleventy),
 * γι' αυτό δεν ορίζουμε ξεχωριστό `dir.layouts`.
 */
export const config = {
  dir: {
    input: "src",
    output: "_site",
    includes: "_includes",
    data: "_data",
  },
  // Ποιες καταλήξεις θεωρούνται templates προς επεξεργασία.
  templateFormats: ["njk", "md", "html"],
  markdownTemplateEngine: "njk",
  htmlTemplateEngine: "njk",
};
