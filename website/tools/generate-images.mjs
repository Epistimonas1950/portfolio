/**
 * Γεννήτρια εικόνων-υποκατάστατων (placeholders).
 *
 * ΓΙΑΤΙ ΥΠΑΡΧΕΙ: δουλεύουμε τοπικά και χωρίς πραγματικές φωτογραφίες, αλλά ένα
 * site χορού χωρίς εικόνες δεν αξιολογείται. Το script φτιάχνει SVG με θέμα τη
 * «φωτογραφία πολλαπλής έκθεσης»: την ίδια φιγούρα σε 5–8 στιγμές μιας κίνησης.
 *
 * ΠΩΣ ΤΡΕΧΕΙ:      npm run images
 * ΠΟΥ ΓΡΑΦΕΙ:      src/assets/img/
 *
 * ΟΤΑΝ ΕΡΘΟΥΝ ΟΙ ΑΛΗΘΙΝΕΣ ΦΩΤΟΓΡΑΦΙΕΣ: σβήνεις τα SVG, βάζεις τα .jpg με τα
 * ίδια ονόματα και αλλάζεις την κατάληξη στα _data/*.js και στο front matter.
 * Το script δεν χρειάζεται ξανά — δεν είναι μέρος του build.
 *
 * Όλα παράγονται ντετερμινιστικά από ένα «seed» (το όνομα του αρχείου), ώστε
 * κάθε μάθημα να έχει πάντα την ίδια εικόνα σε κάθε εκτέλεση.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "src", "assets", "img");

/* =====================================================================
 * 1. Παλέτα — ίδια χρώματα με το CSS (var(--ink), var(--amber) κ.λπ.)
 * ================================================================== */

const INK = "#17101C"; // βαθύ δαμασκηνί, «σβηστά φώτα»
const PLUM = "#2E1D38"; // ένα σκαλί πιο ανοιχτό, για το βάθος
const CHALK = "#F5F1EC"; // κιμωλία / ρετσίνι — η φιγούρα στο προσκήνιο

// Χρώματα «ζελατίνας» προβολέα. Κάθε εικόνα παίρνει ένα, ανάλογα με το seed.
const GELS = ["#E9A13B", "#D8607A", "#7FA8C9", "#C98A5B", "#B98BC4"];

/* =====================================================================
 * 2. Τυχαιότητα με seed (ίδιο seed → ίδια εικόνα, πάντα)
 * ================================================================== */

/** Μετατρέπει ένα κείμενο σε ακέραιο (hash FNV-1a). */
function hash(text) {
  let h = 2166136261;
  for (const char of text) {
    h ^= char.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Ψευδοτυχαίος αριθμός 0–1, ντετερμινιστικός (mulberry32). */
function makeRandom(seed) {
  let state = hash(seed);
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* =====================================================================
 * 3. Η φιγούρα
 *
 * Το «σκελετό» τον ορίζουμε με γωνίες μελών, σε μοίρες:
 *   0° = δεξιά,  90° = κάτω,  -90° = πάνω   (στο SVG ο άξονας y κοιτάει κάτω)
 * Κάθε μέλος έχει δύο γωνίες: [πάνω τμήμα, κάτω τμήμα] — π.χ. βραχίονας/πήχης.
 * ================================================================== */

const TORSO = 40; // μήκος κορμού
const HEAD_R = 7; // ακτίνα κεφαλιού
const UPPER_ARM = 17;
const FOREARM = 17;
const THIGH = 25;
const SHIN = 25;

const POSES = {
  // Όρθια, χέρια χαμηλά — η «ουδέτερη» στάση.
  stand: { lean: 0, armA: [98, 96], armB: [82, 84], legA: [93, 92], legB: [87, 88] },
  // Δεύτερη θέση: χέρια ανοιχτά στο πλάι.
  second: { lean: 0, armA: [168, 176], armB: [12, 4], legA: [108, 103], legB: [72, 77] },
  // Πέμπτη θέση: χέρια ψηλά σε στεφάνι.
  fifth: { lean: -4, armA: [-106, -98], armB: [-74, -82], legA: [95, 93], legB: [85, 87] },
  // Attitude: το ένα πόδι λυγισμένο ψηλά.
  attitude: { lean: -8, armA: [-124, -150], armB: [8, -12], legA: [138, 96], legB: [88, 90] },
  // Αραμπέσκ: κορμός μπροστά, πόδι τεντωμένο πίσω και ελαφρώς ψηλά.
  arabesque: { lean: -32, armA: [-24, -14], armB: [152, 166], legA: [158, 162], legB: [86, 88] },
  // Άλμα σε δεύτερη θέση.
  jump: { lean: -2, armA: [-142, -162], armB: [-38, -18], legA: [128, 148], legB: [52, 32] },
  // Άνοιγμα προς τα πάνω — τυπικό του σύγχρονου.
  reach: { lean: -14, armA: [-98, -94], armB: [-56, -30], legA: [96, 93], legB: [78, 82] },
  // Χαμηλή στάση στο πάτωμα (floor work).
  floor: { lean: -68, armA: [22, 44], armB: [158, 176], legA: [28, 8], legB: [138, 162] },
  // Λαβή ζευγαριού / τάνγκο: κορμός ελαφρώς μπροστά, ένα χέρι τεντωμένο.
  hold: { lean: -10, armA: [-8, -4], armB: [128, 150], legA: [104, 100], legB: [76, 84] },
  // Κλειστή στροφή (pirouette): πόδι σε passé.
  turn: { lean: -2, armA: [-150, -120], armB: [-30, -60], legA: [126, 84], legB: [90, 91] },
};

/* Ποιο ζευγάρι στάσεων ταιριάζει σε κάθε ύφος χορού.
   Η φιγούρα «ταξιδεύει» από την πρώτη στη δεύτερη στάση. */
const MOVES = {
  baleto: ["fifth", "arabesque"],
  sygxronos: ["floor", "reach"],
  "hip-hop": ["second", "jump"],
  latin: ["stand", "turn"],
  tango: ["hold", "second"],
  paradosiakoi: ["hold", "stand"],
  oriental: ["second", "fifth"],
  paidiko: ["stand", "jump"],
  default: ["stand", "reach"],
};

/** Σημείο σε απόσταση `length` από το `from`, σε γωνία `deg`. */
function step([x, y], length, deg) {
  const rad = (deg * Math.PI) / 180;
  return [x + length * Math.cos(rad), y + length * Math.sin(rad)];
}

/** Γραμμική παρεμβολή ανάμεσα σε δύο τιμές. */
const lerp = (a, b, t) => a + (b - a) * t;

/** Παρεμβολή ανάμεσα σε δύο στάσεις — έτσι βγαίνουν τα ενδιάμεσα καρέ. */
function blendPoses(a, b, t) {
  const blendPair = (p, q) => [lerp(p[0], q[0], t), lerp(p[1], q[1], t)];
  return {
    lean: lerp(a.lean, b.lean, t),
    armA: blendPair(a.armA, b.armA),
    armB: blendPair(a.armB, b.armB),
    legA: blendPair(a.legA, b.legA),
    legB: blendPair(a.legB, b.legB),
  };
}

/**
 * Επιστρέφει το SVG μιας φιγούρας.
 * Η αρχή των αξόνων (0,0) είναι η λεκάνη· η φιγούρα «κάθεται» γύρω από εκεί.
 */
function figure(pose, { color, opacity, width }) {
  const hip = [0, 0];
  const shoulder = step(hip, TORSO, -90 + pose.lean);
  const head = step(shoulder, HEAD_R + 5, -90 + pose.lean);

  // Ένα μέλος = δύο τμήματα (π.χ. μηρός + κνήμη).
  const limb = (origin, [a1, a2], l1, l2, tilt = 0) => {
    const joint = step(origin, l1, a1 + tilt);
    const tip = step(joint, l2, a2 + tilt);
    return `${origin[0].toFixed(1)},${origin[1].toFixed(1)} ${joint[0].toFixed(1)},${joint[1].toFixed(1)} ${tip[0].toFixed(1)},${tip[1].toFixed(1)}`;
  };

  const stroke = `stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" fill="none"`;

  return `<g opacity="${opacity.toFixed(3)}">
      <line x1="${hip[0]}" y1="${hip[1]}" x2="${shoulder[0].toFixed(1)}" y2="${shoulder[1].toFixed(1)}" ${stroke}/>
      <circle cx="${head[0].toFixed(1)}" cy="${head[1].toFixed(1)}" r="${HEAD_R}" fill="${color}"/>
      <polyline points="${limb(shoulder, pose.armA, UPPER_ARM, FOREARM, pose.lean)}" ${stroke}/>
      <polyline points="${limb(shoulder, pose.armB, UPPER_ARM, FOREARM, pose.lean)}" ${stroke}/>
      <polyline points="${limb(hip, pose.legA, THIGH, SHIN)}" ${stroke}/>
      <polyline points="${limb(hip, pose.legB, THIGH, SHIN)}" ${stroke}/>
    </g>`;
}

/* =====================================================================
 * 4. Κοινά κομμάτια σκηνικού
 * ================================================================== */

/** Φόντο: κάθετη διαβάθμιση + φωτεινός κώνος προβολέα + κόκκος φιλμ. */
function backdrop(id, width, height, gel, spot) {
  return `<defs>
    <linearGradient id="bg-${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${PLUM}"/>
      <stop offset="70%" stop-color="${INK}"/>
      <stop offset="100%" stop-color="${INK}"/>
    </linearGradient>
    <radialGradient id="spot-${id}" cx="${spot.x}" cy="${spot.y}" r="${spot.r}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="${gel}" stop-opacity="0.42"/>
      <stop offset="55%" stop-color="${gel}" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="${gel}" stop-opacity="0"/>
    </radialGradient>
    <!-- Ο κόκκος φτιάχνεται μία φορά σε μικρό πλακίδιο και επαναλαμβάνεται,
         ώστε το φίλτρο να μην τρέχει σε ολόκληρη την εικόνα. -->
    <filter id="noise-${id}" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="${id}"/>
      <feColorMatrix type="saturate" values="0"/>
    </filter>
    <pattern id="grain-${id}" width="140" height="140" patternUnits="userSpaceOnUse">
      <rect width="140" height="140" filter="url(#noise-${id})" opacity="0.5"/>
    </pattern>
  </defs>
  <rect width="${width}" height="${height}" fill="url(#bg-${id})"/>
  <rect width="${width}" height="${height}" fill="url(#spot-${id})"/>`;
}

/** Ο κόκκος μπαίνει τελευταίος, πάνω από όλα. */
const grainOverlay = (id, width, height) =>
  `<rect width="${width}" height="${height}" fill="url(#grain-${id})" opacity="0.07" style="mix-blend-mode:overlay"/>`;

/** Το δάπεδο: μια απαλή οριζόντια ταινία φωτός. */
function floorLine(width, y, gel) {
  return `<g opacity="0.5">
    <rect x="0" y="${y}" width="${width}" height="1.5" fill="${gel}" opacity="0.35"/>
    <rect x="0" y="${y + 1.5}" width="${width}" height="${Math.max(0, 60)}" fill="${INK}" opacity="0.35"/>
  </g>`;
}

/* =====================================================================
 * 5. Οι τρεις τύποι εικόνας
 * ================================================================== */

/**
 * «Πολλαπλή έκθεση»: η ίδια φιγούρα σε διαδοχικές στιγμές μιας κίνησης.
 * Χρησιμοποιείται για τα μαθήματα, τα νέα και τη μισή γκαλερί.
 */
function strobe(seed, width, height, { move = "default", frames = 7 } = {}) {
  const random = makeRandom(seed);
  const gel = GELS[Math.floor(random() * GELS.length)];
  const [from, to] = MOVES[move] ?? MOVES.default;

  /*
   * Διαστασιολόγηση: με τα χέρια ψηλά η φιγούρα πιάνει ~74 μονάδες πάνω από τη
   * λεκάνη και 50 κάτω (τεντωμένο πόδι) — σύνολο ~124. Κλιμακώνουμε ώστε να
   * καταλαμβάνει το 56% του ύψους και να μένει αέρας πάνω από τα χέρια.
   */
  const scale = (height * 0.56) / 124;
  const baseline = height * 0.84; // ύψος δαπέδου
  const startX = width * 0.16;
  const endX = width * 0.79;

  const bodies = [];
  for (let i = 0; i < frames; i++) {
    const t = frames === 1 ? 1 : i / (frames - 1);
    const last = i === frames - 1;

    const pose = blendPoses(POSES[from], POSES[to], t * t * (3 - 2 * t)); // ease
    const x = lerp(startX, endX, t);
    const bob = Math.sin(t * Math.PI) * height * 0.035; // μικρή «ανάσα» στο άλμα
    const y = baseline - bob - THIGH * scale - SHIN * scale + 6 * scale;
    const s = scale * lerp(0.9, 1.05, t);

    bodies.push(
      `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) scale(${s.toFixed(3)})">` +
        figure(pose, {
          color: last ? CHALK : gel,
          opacity: last ? 0.96 : 0.1 + 0.42 * t,
          width: last ? 6 : 5,
        }) +
        `</g>`
    );
  }

  return svg(seed, width, height, [
    backdrop(hash(seed) % 9999, width, height, gel, {
      x: endX,
      y: baseline - height * 0.3,
      r: height * 0.62,
    }),
    floorLine(width, baseline, gel),
    // Σκιά κάτω από την τελευταία φιγούρα.
    `<ellipse cx="${endX.toFixed(1)}" cy="${baseline.toFixed(1)}" rx="${(height * 0.11).toFixed(1)}" ry="${(height * 0.016).toFixed(1)}" fill="${INK}" opacity="0.55"/>`,
    bodies.join("\n    "),
    grainOverlay(hash(seed) % 9999, width, height),
  ]);
}

/**
 * «Πορτρέτο σε προβολέα»: μία μόνο φιγούρα, κεντραρισμένη, σε δέσμη φωτός.
 * Χρησιμοποιείται για τους καθηγητές. Η διαφορά από τα μαθήματα είναι
 * σκόπιμη: το μάθημα είναι κίνηση (πολλές φιγούρες), ο καθηγητής πρόσωπο (μία).
 */
function portrait(seed, width, height) {
  const random = makeRandom(seed);
  const gel = GELS[Math.floor(random() * GELS.length)];
  const id = hash(seed) % 9999;

  // Κάθε καθηγητής παίρνει σταθερά τη δική του στάση, ανάλογα με το seed.
  const options = ["fifth", "second", "reach", "hold", "attitude", "turn"];
  const pose = POSES[options[Math.floor(random() * options.length)]];

  const scale = (height * 0.66) / 124;
  const baseline = height * 0.9;
  const cx = width * 0.5;
  const hipY = baseline - 44 * scale;

  return svg(seed, width, height, [
    backdrop(id, width, height, gel, { x: cx, y: hipY - 30 * scale, r: height * 0.6 }),
    // Δέσμη προβολέα από ψηλά: ένα τραπέζιο που ανοίγει προς το δάπεδο.
    `<path d="M ${cx - width * 0.1} 0 L ${cx + width * 0.1} 0 L ${cx + width * 0.42} ${baseline} L ${cx - width * 0.42} ${baseline} Z"
           fill="${gel}" opacity="0.07"/>`,
    `<ellipse cx="${cx}" cy="${baseline}" rx="${width * 0.34}" ry="${height * 0.035}" fill="${gel}" opacity="0.12"/>`,
    `<ellipse cx="${cx}" cy="${baseline}" rx="${width * 0.14}" ry="${height * 0.016}" fill="${INK}" opacity="0.6"/>`,
    `<g transform="translate(${cx} ${hipY.toFixed(1)}) scale(${scale.toFixed(3)})">` +
      figure(pose, { color: CHALK, opacity: 0.93, width: 6 }) +
      `</g>`,
    floorLine(width, baseline, gel),
    grainOverlay(id, width, height),
  ]);
}

/**
 * «Άδεια αίθουσα»: καθρέφτης, μπάρα, δάπεδο.
 * Για τις φωτογραφίες χώρου στη γκαλερί και στη σελίδα «Η σχολή».
 */
function room(seed, width, height) {
  const random = makeRandom(seed);
  const gel = GELS[Math.floor(random() * GELS.length)];
  const id = hash(seed) % 9999;

  const floorY = height * 0.74;
  const mirrorTop = height * 0.14;
  const mirrorLeft = width * 0.07;
  const mirrorWidth = width * 0.86;
  const mirrorHeight = floorY - height * 0.19;

  // Μια μοναχική φιγούρα δίνει κλίμακα στον χώρο — αλλιώς το κάδρο διαβάζεται
  // σαν άδειο ορθογώνιο και όχι σαν αίθουσα.
  const figureScale = (height * 0.44) / 124;
  const figureX = width * 0.72;

  return svg(seed, width, height, [
    backdrop(id, width, height, gel, { x: width * 0.68, y: height * 0.3, r: height * 1.05 }),
    `<defs>
      <linearGradient id="mirror-${id}" x1="0" y1="0" x2="0.35" y2="1">
        <stop offset="0%" stop-color="${CHALK}" stop-opacity="0.10"/>
        <stop offset="45%" stop-color="${PLUM}" stop-opacity="0.55"/>
        <stop offset="100%" stop-color="${INK}" stop-opacity="0.75"/>
      </linearGradient>
    </defs>`,
    // Ο καθρέφτης καταλαμβάνει σχεδόν όλον τον τοίχο, όπως σε κάθε αίθουσα χορού.
    `<rect x="${mirrorLeft}" y="${mirrorTop}" width="${mirrorWidth}" height="${mirrorHeight}" fill="url(#mirror-${id})"/>`,
    `<rect x="${mirrorLeft}" y="${mirrorTop}" width="${mirrorWidth}" height="${mirrorHeight}"
           fill="none" stroke="${gel}" stroke-width="1.5" opacity="0.4"/>`,
    // Δύο λοξές ανταύγειες πάνω στο τζάμι.
    `<g opacity="0.06" fill="${CHALK}">
      <path d="M ${width * 0.1} ${mirrorTop + mirrorHeight} L ${width * 0.36} ${mirrorTop} L ${width * 0.45} ${mirrorTop} L ${width * 0.19} ${mirrorTop + mirrorHeight} Z"/>
      <path d="M ${width * 0.48} ${mirrorTop + mirrorHeight} L ${width * 0.74} ${mirrorTop} L ${width * 0.77} ${mirrorTop} L ${width * 0.51} ${mirrorTop + mirrorHeight} Z"/>
    </g>`,
    // Λάμπες οροφής: τρεις φωτεινές ταινίες πάνω από τον καθρέφτη.
    `<g fill="${gel}" opacity="0.55">
      <rect x="${width * 0.14}" y="${height * 0.06}" width="${width * 0.16}" height="${Math.max(2, height * 0.006)}" rx="2"/>
      <rect x="${width * 0.42}" y="${height * 0.06}" width="${width * 0.16}" height="${Math.max(2, height * 0.006)}" rx="2"/>
      <rect x="${width * 0.7}" y="${height * 0.06}" width="${width * 0.16}" height="${Math.max(2, height * 0.006)}" rx="2"/>
    </g>`,
    // Η μπάρα, μπροστά από τον καθρέφτη, με τα στηρίγματά της.
    `<g>
      <rect x="${width * 0.05}" y="${floorY - height * 0.21}" width="${width * 0.9}" height="${Math.max(3, height * 0.009)}"
            rx="${height * 0.0045}" fill="${gel}" opacity="0.55"/>
      <rect x="${width * 0.16}" y="${floorY - height * 0.2}" width="2" height="${height * 0.2}" fill="${gel}" opacity="0.2"/>
      <rect x="${width * 0.82}" y="${floorY - height * 0.2}" width="2" height="${height * 0.2}" fill="${gel}" opacity="0.2"/>
    </g>`,
    // Η φιγούρα, ακουμπισμένη στη μπάρα σε ζέσταμα.
    `<g transform="translate(${figureX.toFixed(1)} ${(floorY - 44 * figureScale).toFixed(1)}) scale(${figureScale.toFixed(3)})">` +
      figure(blendPoses(POSES.stand, POSES.hold, 0.55), { color: CHALK, opacity: 0.5, width: 5.5 }) +
      `</g>`,
    floorLine(width, floorY, gel),
    // Το γυαλιστερό δάπεδο επιστρέφει θολά ό,τι είναι από πάνω.
    `<g opacity="0.14">
      <rect x="${width * 0.05}" y="${floorY + height * 0.055}" width="${width * 0.9}" height="1.5" fill="${gel}"/>
      <ellipse cx="${figureX}" cy="${floorY + height * 0.03}" rx="${height * 0.045}" ry="${height * 0.012}" fill="${CHALK}"/>
    </g>`,
    grainOverlay(id, width, height),
  ]);
}

/** Τυλίγει τα κομμάτια σε ολοκληρωμένο SVG αρχείο. */
function svg(seed, width, height, parts) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">
  <title>${seed}</title>
  ${parts.join("\n  ")}
</svg>
`;
}

/* =====================================================================
 * 6. Λογότυπο και favicon
 *
 * Το σήμα είναι ένα «8» φτιαγμένο από δύο κύκλους σε ελαφριά κλίση: το
 * μέτρημα των οκτώ, αλλά και μια φιγούρα σε κίνηση.
 * ================================================================== */

const logoMark = (stroke, width = 6) =>
  `<g transform="rotate(-14 24 24)" fill="none" stroke="${stroke}" stroke-width="${width}">
      <circle cx="24" cy="15" r="9.5"/>
      <circle cx="24" cy="33" r="11.5"/>
    </g>`;

const logoSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48" height="48" role="img">
  <title>ΟΚΤΩ</title>
  ${logoMark("currentColor")}
</svg>
`;

const faviconSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48" height="48" role="img">
  <title>ΟΚΤΩ</title>
  <rect width="48" height="48" rx="10" fill="${INK}"/>
  <g transform="scale(0.82) translate(5.3 5.3)">${logoMark("#E9A13B", 5.5)}</g>
</svg>
`;

/* =====================================================================
 * 7. Ποιες εικόνες φτιάχνονται
 * ================================================================== */

const CLASS_SLUGS = [
  "baleto",
  "sygxronos",
  "hip-hop",
  "latin",
  "tango",
  "paradosiakoi",
  "oriental",
  "paidiko",
];

const TEACHER_SLUGS = [
  "eleni-papadaki",
  "nikos-vlachakis",
  "marina-kostea",
  "thanasis-rigas",
  "dafni-antoniou",
  "zoi-markaki",
];

const NEWS_SLUGS = [
  "eggrafes-2026",
  "seminario-bachata",
  "parastasi-iouniou",
  "neo-tmima-tango",
];

/* Η γκαλερί εναλλάσσει τους τρεις τύπους εικόνας ώστε να μη μοιάζουν όλες ίδιες.
   Η σειρά αντιστοιχεί ένα προς ένα στο src/_data/gallery.js. */
const GALLERY = [
  { file: "g-01.svg", type: "room", shape: "landscape" },
  { file: "g-02.svg", type: "strobe", shape: "portrait", move: "baleto" },
  { file: "g-03.svg", type: "strobe", shape: "portrait", move: "tango" },
  { file: "g-04.svg", type: "strobe", shape: "landscape", move: "hip-hop" },
  { file: "g-05.svg", type: "strobe", shape: "portrait", move: "paidiko" },
  { file: "g-06.svg", type: "strobe", shape: "portrait", move: "sygxronos" },
  { file: "g-07.svg", type: "room", shape: "landscape" },
  { file: "g-08.svg", type: "strobe", shape: "portrait", move: "paradosiakoi" },
  { file: "g-09.svg", type: "strobe", shape: "portrait", move: "latin" },
  { file: "g-10.svg", type: "room", shape: "landscape" },
  { file: "g-11.svg", type: "portrait", shape: "portrait" },
  { file: "g-12.svg", type: "strobe", shape: "portrait", move: "oriental" },
];

async function write(relativePath, contents) {
  const target = join(OUT, relativePath);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, contents, "utf8");
  return relativePath;
}

async function main() {
  const written = [];

  // Μαθήματα — κάθετες εικόνες 4:5.
  for (const slug of CLASS_SLUGS) {
    written.push(
      await write(`classes/${slug}.svg`, strobe(slug, 900, 1125, { move: slug, frames: 6 }))
    );
  }

  // Καθηγητές — πορτρέτα 4:5.
  for (const slug of TEACHER_SLUGS) {
    written.push(await write(`teachers/${slug}.svg`, portrait(slug, 800, 1000)));
  }

  // Νέα — πλατιές εικόνες 16:9.
  for (const [index, slug] of NEWS_SLUGS.entries()) {
    const move = CLASS_SLUGS[(index * 3) % CLASS_SLUGS.length];
    written.push(await write(`news/${slug}.svg`, strobe(slug, 1200, 675, { move, frames: 8 })));
  }

  // Γκαλερί.
  for (const item of GALLERY) {
    const [w, h] = item.shape === "landscape" ? [1200, 900] : [800, 1200];
    const seed = `gallery-${item.file}`;
    const image =
      item.type === "room"
        ? room(seed, w, h)
        : item.type === "portrait"
          ? portrait(seed, w, h)
          : strobe(seed, w, h, { move: item.move, frames: item.shape === "landscape" ? 8 : 5 });
    written.push(await write(`gallery/${item.file}`, image));
  }

  // Μεγάλες εικόνες σελίδων.
  written.push(await write("hero.svg", strobe("hero-okto", 1400, 1000, { move: "sygxronos", frames: 8 })));
  written.push(await write("studio.svg", room("studio-a", 1400, 900)));
  written.push(await write("logo.svg", logoSvg));
  written.push(await write("favicon.svg", faviconSvg));

  console.log(`Γράφτηκαν ${written.length} αρχεία στο src/assets/img/`);
}

main();
