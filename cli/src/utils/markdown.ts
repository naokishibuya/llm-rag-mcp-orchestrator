import chalk from "chalk";
import { createLowlight, common } from "lowlight";

const lowlight = createLowlight(common);

// Theme mapping for highlight.js scopes to chalk styles
const THEME: Record<string, (s: string) => string> = {
  keyword: chalk.magenta,
  built_in: chalk.cyan,
  string: chalk.green,
  number: chalk.yellow,
  literal: chalk.yellow,
  comment: chalk.gray,
  doctag: chalk.gray,
  title: chalk.blue.bold,
  "title.function": chalk.blue,
  "title.class": chalk.blue.bold,
  params: chalk.white,
  attr: chalk.cyan,
  attribute: chalk.cyan,
  selector: chalk.yellow,
  symbol: chalk.red,
  bullet: chalk.red,
  addition: chalk.green,
  deletion: chalk.red,
  meta: chalk.gray,
  type: chalk.cyan,
  name: chalk.blue,
  tag: chalk.blue,
  variable: chalk.red,
  "variable.language": chalk.red,
  regexp: chalk.red,
  link: chalk.cyan.underline,
  operator: chalk.white,
  punctuation: chalk.white,
};

function applyStyle(className: string, text: string): string {
  const styleFn = THEME[className];
  return styleFn ? styleFn(text) : text;
}

// Recursively render hast nodes from lowlight into chalk-styled strings
function renderHastNodes(nodes: any[]): string {
  let result = "";
  for (const node of nodes) {
    if (node.type === "text") {
      result += node.value;
    } else if (node.type === "element") {
      const classes: string[] = node.properties?.className || [];
      const inner = renderHastNodes(node.children || []);
      // hljs classes are like "hljs-keyword", strip the prefix
      const scope = classes.map((c: string) => c.replace("hljs-", "")).find((c: string) => THEME[c]);
      result += scope ? applyStyle(scope, inner) : inner;
    }
  }
  return result;
}

export function highlightCode(code: string, lang?: string): string {
  try {
    const tree = lang && lowlight.listLanguages().includes(lang)
      ? lowlight.highlight(lang, code)
      : lowlight.highlightAuto(code);
    return renderHastNodes(tree.children);
  } catch {
    return code;
  }
}

/**
 * Render markdown text into chalk-styled terminal output.
 * Handles: headers, code blocks, math blocks, inline code/math, bold, italic, links, lists, blockquotes, horizontal rules.
 */
export function renderMarkdown(text: string): string {
  const lines = text.split("\n");
  const output: string[] = [];
  let inCodeBlock = false;
  let codeLang = "";
  let codeLines: string[] = [];
  let inMathBlock = false;
  let mathLines: string[] = [];
  let tableRows: string[] = [];

  const flushTable = () => {
    if (tableRows.length === 0) return;
    output.push(...renderTable(tableRows));
    tableRows = [];
  };

  for (const line of lines) {
    // Code block fences
    const fenceMatch = line.match(/^```(\w*)/);
    if (fenceMatch && !inMathBlock) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLang = fenceMatch[1];
        codeLines = [];
      } else {
        // End of code block
        const code = codeLines.join("\n");
        const highlighted = highlightCode(code, codeLang);
        const langLabel = codeLang ? chalk.dim(` ${codeLang} `) : "";
        output.push(chalk.dim("┌─") + langLabel + chalk.dim("─".repeat(Math.max(0, 40 - langLabel.length))));
        for (const hl of highlighted.split("\n")) {
          output.push(chalk.dim("│ ") + hl);
        }
        output.push(chalk.dim("└" + "─".repeat(42)));
        inCodeBlock = false;
        codeLang = "";
        codeLines = [];
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // Display math block: $$ on its own line
    if (line.trim() === "$$") {
      if (!inMathBlock) {
        inMathBlock = true;
        mathLines = [];
      } else {
        // End of math block
        const math = mathLines.join("\n");
        output.push(chalk.dim("┌─") + chalk.magenta(" math ") + chalk.dim("─".repeat(35)));
        for (const ml of math.split("\n")) {
          output.push(chalk.dim("│ ") + chalk.magenta(renderLatex(ml)));
        }
        output.push(chalk.dim("└" + "─".repeat(42)));
        inMathBlock = false;
        mathLines = [];
      }
      continue;
    }

    if (inMathBlock) {
      mathLines.push(line);
      continue;
    }

    // Table rows: lines with | (either |col|col| or col|col format)
    const pipeCount = (line.match(/\|/g) || []).length;
    if (pipeCount >= 1 && (
      /^\|/.test(line) ||                           // leading pipe: | col | col |
      /^[\s:]*-{2,}[\s:]*\|/.test(line) ||          // separator: ---|---
      pipeCount >= 2 ||                              // multiple pipes: col | col | col
      tableRows.length > 0                           // continuation of existing table
    )) {
      tableRows.push(line);
      continue;
    }
    flushTable();

    // Headers
    const headerMatch = line.match(/^(#{1,6})\s+(.*)/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      const content = renderInline(headerMatch[2]);
      if (level === 1) {
        output.push(chalk.bold.underline(content));
      } else if (level === 2) {
        output.push(chalk.bold(content));
      } else {
        output.push(chalk.bold.dim(content));
      }
      continue;
    }

    // Single-line display math: $$...$$ on one line
    const inlineDisplayMath = line.match(/^\$\$(.+)\$\$$/);
    if (inlineDisplayMath) {
      const math = inlineDisplayMath[1].trim();
      output.push(chalk.dim("  ") + chalk.magenta(renderLatex(math)));
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line)) {
      output.push(chalk.dim("─".repeat(42)));
      continue;
    }

    // Blockquote
    const bqMatch = line.match(/^>\s?(.*)/);
    if (bqMatch) {
      output.push(chalk.dim("│ ") + chalk.italic(renderInline(bqMatch[1])));
      continue;
    }

    // Unordered list
    const ulMatch = line.match(/^(\s*)[-*+]\s+(.*)/);
    if (ulMatch) {
      const indent = ulMatch[1];
      output.push(indent + chalk.dim("  • ") + renderInline(ulMatch[2]));
      continue;
    }

    // Ordered list
    const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)/);
    if (olMatch) {
      const indent = olMatch[1];
      output.push(indent + chalk.dim(`  ${olMatch[2]}. `) + renderInline(olMatch[3]));
      continue;
    }

    // Regular paragraph
    output.push(renderInline(line));
  }

  // Flush any remaining table
  flushTable();

  // Handle unclosed code block
  if (inCodeBlock && codeLines.length > 0) {
    const code = codeLines.join("\n");
    const highlighted = highlightCode(code, codeLang);
    output.push(chalk.dim("┌─") + chalk.dim("─".repeat(40)));
    for (const hl of highlighted.split("\n")) {
      output.push(chalk.dim("│ ") + hl);
    }
    output.push(chalk.dim("└" + "─".repeat(42)));
  }

  // Handle unclosed math block
  if (inMathBlock && mathLines.length > 0) {
    const math = mathLines.join("\n");
    output.push(chalk.dim("┌─") + chalk.magenta(" math ") + chalk.dim("─".repeat(35)));
    for (const ml of math.split("\n")) {
      output.push(chalk.dim("│ ") + chalk.magenta(renderLatex(ml)));
    }
    output.push(chalk.dim("└" + "─".repeat(42)));
  }

  return output.join("\n");
}

/** Render a markdown table (array of raw table lines) into box-drawn output */
function renderTable(rows: string[]): string[] {
  // Skip separator rows like |---|---| or ---|---|---
  const isSeparator = (r: string) => /^[\s|]*[-:]+[\s|]*(\|[\s]*[-:]+[\s|]*)+$/.test(r);

  // Parse cells: handle both |col|col| and col|col formats
  const parseCells = (r: string): string[] => {
    const trimmed = r.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      return trimmed.split("|").slice(1, -1).map((c) => c.trim());
    }
    if (trimmed.startsWith("|")) {
      return trimmed.split("|").slice(1).map((c) => c.trim());
    }
    return trimmed.split("|").map((c) => c.trim());
  };

  const parsed = rows.filter((r) => !isSeparator(r)).map(parseCells);

  if (parsed.length === 0) return [];

  const colCount = Math.max(...parsed.map((r) => r.length));

  // Normalize all rows to same column count
  const normalized = parsed.map((r) => {
    while (r.length < colCount) r.push("");
    return r;
  });

  // Strip chalk formatting to calculate visible widths
  const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

  // Render inline markdown for each cell, then compute widths
  const rendered = normalized.map((r) => r.map((c) => renderInline(c)));
  const colWidths = Array.from({ length: colCount }, (_, i) =>
    Math.max(3, ...rendered.map((r) => stripAnsi(r[i] || "").length)),
  );

  // Pad a cell to its column width (accounting for ANSI codes)
  const pad = (cell: string, width: number) => {
    const visible = stripAnsi(cell).length;
    return cell + " ".repeat(Math.max(0, width - visible));
  };

  const hline = (left: string, mid: string, right: string, fill: string) =>
    chalk.dim(
      left +
        colWidths.map((w) => fill.repeat(w + 2)).join(mid) +
        right,
    );

  const output: string[] = [];
  output.push(hline("┌", "┬", "┐", "─"));

  rendered.forEach((row, ri) => {
    const cells = row.map((c, ci) => ` ${pad(c, colWidths[ci])} `);
    output.push(chalk.dim("│") + cells.join(chalk.dim("│")) + chalk.dim("│"));

    if (ri === 0 && rendered.length > 1) {
      // Header separator (bold line)
      output.push(hline("├", "┼", "┤", "─"));
    }
  });

  output.push(hline("└", "┴", "┘", "─"));
  return output;
}

/**
 * Convert common LaTeX commands to Unicode approximations for terminal display.
 * This isn't full LaTeX rendering, but makes expressions more readable.
 */
function renderLatex(tex: string): string {
  // Remove escaped dollar signs from backend normalization
  let t = tex.replace(/\\\$/g, "$");

  // Common symbols
  const symbols: [RegExp, string][] = [
    [/\\alpha/g, "α"], [/\\beta/g, "β"], [/\\gamma/g, "γ"], [/\\delta/g, "δ"],
    [/\\epsilon/g, "ε"], [/\\zeta/g, "ζ"], [/\\eta/g, "η"], [/\\theta/g, "θ"],
    [/\\lambda/g, "λ"], [/\\mu/g, "μ"], [/\\pi/g, "π"], [/\\sigma/g, "σ"],
    [/\\tau/g, "τ"], [/\\phi/g, "φ"], [/\\omega/g, "ω"],
    [/\\Alpha/g, "Α"], [/\\Beta/g, "Β"], [/\\Gamma/g, "Γ"], [/\\Delta/g, "Δ"],
    [/\\Theta/g, "Θ"], [/\\Lambda/g, "Λ"], [/\\Pi/g, "Π"], [/\\Sigma/g, "Σ"],
    [/\\Phi/g, "Φ"], [/\\Omega/g, "Ω"],
    [/\\infty/g, "∞"], [/\\pm/g, "±"], [/\\mp/g, "∓"],
    [/\\times/g, "×"], [/\\div/g, "÷"], [/\\cdot/g, "·"], [/\\ldots/g, "…"], [/\\cdots/g, "⋯"],
    [/\\leq/g, "≤"], [/\\geq/g, "≥"], [/\\neq/g, "≠"], [/\\approx/g, "≈"],
    [/\\equiv/g, "≡"], [/\\sim/g, "∼"],
    [/\\in/g, "∈"], [/\\notin/g, "∉"], [/\\subset/g, "⊂"], [/\\supset/g, "⊃"],
    [/\\subseteq/g, "⊆"], [/\\supseteq/g, "⊇"],
    [/\\cup/g, "∪"], [/\\cap/g, "∩"],
    [/\\forall/g, "∀"], [/\\exists/g, "∃"],
    [/\\nabla/g, "∇"], [/\\partial/g, "∂"],
    [/\\int/g, "∫"], [/\\sum/g, "∑"], [/\\prod/g, "∏"],
    [/\\to/g, "→"], [/\\rightarrow/g, "→"], [/\\leftarrow/g, "←"],
    [/\\Rightarrow/g, "⇒"], [/\\Leftarrow/g, "⇐"], [/\\iff/g, "⇔"],
    [/\\sqrt\{([^}]+)\}/g, "√($1)"],
    [/\\frac\{([^}]+)\}\{([^}]+)\}/g, "($1)/($2)"],
    [/\\text\{([^}]+)\}/g, "$1"],
    [/\\mathrm\{([^}]+)\}/g, "$1"],
    [/\\mathbf\{([^}]+)\}/g, "$1"],
    [/\\left/g, ""], [/\\right/g, ""],
    [/\\[,;!]/g, " "],
    [/[{}]/g, ""],
    [/\^(\w)/g, "^$1"], [/_(\w)/g, "_$1"],
  ];

  for (const [re, rep] of symbols) {
    t = t.replace(re, rep);
  }
  return t;
}

/** Render inline markdown: bold, italic, inline code, inline math, links, strikethrough */
function renderInline(text: string): string {
  // Protect escaped dollar signs \$ from being consumed by the math regex
  const DOLLAR_PLACEHOLDER = "\x00DOLLAR\x00";
  text = text.replace(/\\\$/g, DOLLAR_PLACEHOLDER);
  // Inline code (must be before bold/italic to avoid conflicts)
  text = text.replace(/`([^`]+)`/g, (_, code) => chalk.bgGray.white(` ${code} `));
  // Inline math $...$ (but not display $$)
  text = text.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_, math) =>
    chalk.magenta(renderLatex(math)));
  // Restore escaped dollar signs
  text = text.replaceAll(DOLLAR_PLACEHOLDER, "$");
  // Bold + italic
  text = text.replace(/\*\*\*(.+?)\*\*\*/g, (_, t) => chalk.bold.italic(t));
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, (_, t) => chalk.bold(t));
  text = text.replace(/__(.+?)__/g, (_, t) => chalk.bold(t));
  // Italic
  text = text.replace(/\*(.+?)\*/g, (_, t) => chalk.italic(t));
  text = text.replace(/_(.+?)_/g, (_, t) => chalk.italic(t));
  // Strikethrough
  text = text.replace(/~~(.+?)~~/g, (_, t) => chalk.strikethrough(t));
  // Links [text](url)
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, linkText, url) =>
    chalk.cyan.underline(linkText) + chalk.dim(` (${url})`));
  return text;
}
