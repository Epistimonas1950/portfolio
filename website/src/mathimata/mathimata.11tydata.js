/**
 * Ρυθμίσεις που ισχύουν για ΟΛΑ τα αρχεία μέσα στο src/mathimata/.
 * (Το Eleventy διαβάζει αυτόματα αρχεία με κατάληξη .11tydata.js.)
 *
 * Έτσι κάθε μάθημα γράφει στο front matter του μόνο ό,τι είναι δικό του —
 * τίτλο, μετρικό, επίπεδα — και όχι το layout ή τα tags.
 */

export default {
  layout: "layouts/class.njk",
  tags: ["class"], // φτιάχνει τη συλλογή collections.classes
};
