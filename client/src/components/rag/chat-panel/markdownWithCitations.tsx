import { Children, isValidElement, useContext, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { FileText, ImageIcon, Brain, Copy, ClipboardCheck } from "lucide-react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import { cn } from "@/lib/utils";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useThemeStore } from "@/stores/useThemeStore";
import type { ChatImageRef, ChatSourceChunk } from "@/types";
import { AllSourcesCtx, useFindDoc } from "./context";

SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("css", css);
SyntaxHighlighter.registerLanguage("html", markup);
SyntaxHighlighter.registerLanguage("xml", markup);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("yml", yaml);
SyntaxHighlighter.registerLanguage("java", java);
SyntaxHighlighter.registerLanguage("go", go);
SyntaxHighlighter.registerLanguage("cpp", cpp);
SyntaxHighlighter.registerLanguage("c", cpp);
SyntaxHighlighter.registerLanguage("diff", diff);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("md", markdown);

function shortenDocName(filename: string, maxLen = 14): string {
    const name = filename.replace(/\.[^.]+$/, "");
    if (name.length <= maxLen) {
        return name;
    }

    return `${name.slice(0, maxLen - 1)}...`;
}

function CitationLink({ index, source, relatedEntities }: { index: string; source: ChatSourceChunk; relatedEntities: string[] }) {
    const { activateCitation, activateCitationKG } = useWorkspaceStore();
    const doc = useFindDoc(source.document_id);

    const isKG = source.source_type === "kg";

    const handleContentClick = () => {
        if (isKG) {
            activateCitationKG(source, relatedEntities, doc);
            return;
        }

        activateCitation(source, relatedEntities, doc);
    };

    const handleKGClick = () => {
        activateCitationKG(source, relatedEntities, doc);
    };

    if (isKG) {
        return (
            <button
                onClick={handleContentClick}
                className="inline-flex items-center gap-0.5 h-[18px] px-1.5 mx-0.5 text-[10px] font-medium rounded-full bg-purple-400/15 text-purple-500 dark:text-purple-400 hover:bg-purple-400/25 transition-colors align-middle whitespace-nowrap"
                title="Xem trong Knowledge Graph"
            >
                <Brain className="w-2.5 h-2.5 flex-shrink-0" />
                <span>KG-{index}</span>
            </button>
        );
    }

    const docName = doc?.original_filename ? shortenDocName(doc.original_filename) : `Nguon ${index}`;
    const label = `${docName}-P.${source.page_no || "?"}`;

    return (
        <span className="inline-flex gap-0.5 mx-0.5 align-middle">
            <button
                onClick={handleContentClick}
                className="inline-flex items-center gap-0.5 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-primary/12 text-primary hover:bg-primary/20 transition-colors whitespace-nowrap"
                title={`Xem nguồn: ${doc?.original_filename || "không rõ"} (p.${source.page_no})`}
            >
                <FileText className="w-2.5 h-2.5 flex-shrink-0" />
                <span>{label}</span>
            </button>
            <button
                onClick={handleKGClick}
                className="inline-flex items-center justify-center w-[18px] h-[18px] text-[10px] font-bold rounded-full bg-purple-400/15 text-purple-500 dark:text-purple-400 hover:bg-purple-400/25 transition-colors"
                title="To sang trong Knowledge Graph"
            >
                <Brain className="w-2.5 h-2.5" />
            </button>
        </span>
    );
}

function InlineImageRef({ imgRefId, imageRef }: { imgRefId: string; imageRef: ChatImageRef }) {
    const [showPreview, setShowPreview] = useState(false);
    const { activateImageCitation } = useWorkspaceStore();
    const doc = useFindDoc(imageRef.document_id);

    const handleClick = () => {
        setShowPreview((prev) => !prev);
        activateImageCitation(imageRef, doc);
    };

    const docName = doc?.original_filename ? shortenDocName(doc.original_filename) : `Hinh ${imgRefId}`;
    const label = `${docName}-P.${imageRef.page_no || "?"}`;

    return (
        <span className="inline-flex flex-col mx-0.5">
            <button
                onClick={handleClick}
                className="inline-flex items-center gap-0.5 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-emerald-400/15 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-400/25 transition-colors align-middle whitespace-nowrap"
                title={imageRef.caption || `Hình từ trang ${imageRef.page_no}`}
            >
                <ImageIcon className="w-2.5 h-2.5 flex-shrink-0" />
                <span>{label}</span>
            </button>
            {showPreview && (
                <a href={imageRef.url} target="_blank" rel="noopener noreferrer" className="block mt-1 rounded-md overflow-hidden border bg-white max-w-[280px] hover:border-primary/50 transition-colors">
                    <img src={imageRef.url} alt={imageRef.caption || `Hình từ trang ${imageRef.page_no}`} className="w-full h-auto max-h-[180px] object-contain" />
                    {imageRef.caption && (
                        <span className="block px-2 py-1 text-[9px] text-muted-foreground leading-tight border-t bg-muted/30">
                            p.{imageRef.page_no} - {imageRef.caption}
                        </span>
                    )}
                </a>
            )}
        </span>
    );
}

const CITATION_RE = /(\[(?:[a-z0-9]+|IMG-[a-z0-9]+)(?:,\s*(?:[a-z0-9]+|IMG-[a-z0-9]+))*\])/g;

function injectCitations(children: ReactNode, sources: ChatSourceChunk[], relatedEntities: string[], imageRefs?: ChatImageRef[], fallbackSources?: ChatSourceChunk[]): ReactNode {
    return Children.map(children, (child) => {
        if (typeof child === "string") {
            const parts = child.split(CITATION_RE);
            if (parts.length === 1) {
                return child;
            }

            const result: ReactNode[] = [];
            parts.forEach((part, index) => {
                const bracketMatch = part.match(/^\[(.+)\]$/);
                if (!bracketMatch) {
                    if (part) {
                        result.push(part);
                    }
                    return;
                }

                const tokens = bracketMatch[1].split(/,\s*/);
                tokens.forEach((token, tokenIndex) => {
                    const key = `${index}-${tokenIndex}`;
                    const imgMatch = token.match(/^IMG-(.+)$/);

                    if (imgMatch && imageRefs && imageRefs.length > 0) {
                        const imgId = imgMatch[1];
                        const imageRef = imageRefs.find((ref) => ref.ref_id === imgId) ?? imageRefs[parseInt(imgId, 10) - 1];
                        if (imageRef) {
                            result.push(<InlineImageRef key={key} imgRefId={imgId} imageRef={imageRef} />);
                            return;
                        }
                    }

                    const source = sources.find((item) => String(item.index) === token) ?? fallbackSources?.find((item) => String(item.index) === token);
                    if (source) {
                        result.push(<CitationLink key={key} index={String(source.index)} source={source} relatedEntities={relatedEntities} />);
                        return;
                    }

                    result.push(`[${token}]`);
                });
            });

            return result;
        }

        if (isValidElement(child) && child.props && (child.props as { children?: ReactNode }).children) {
            const props = child.props as { children?: ReactNode };
            return Object.assign({}, child, {
                props: {
                    ...child.props,
                    children: injectCitations(props.children, sources, relatedEntities, imageRefs, fallbackSources),
                },
            });
        }

        return child;
    });
}

function preprocessMarkdown(text: string): string {
    const lines = text.split("\n");
    const result: string[] = [];
    let prevWasTable = false;
    let inCodeFence = false;

    for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed.startsWith("```")) {
            inCodeFence = !inCodeFence;
        }

        const isTable = (trimmed.startsWith("|") && trimmed.endsWith("|")) || /^\|[\s:|-]+\|$/.test(trimmed);

        if (prevWasTable && !isTable && trimmed !== "") {
            result.push("");
        }

        if (!inCodeFence && trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4 && trimmed !== "$$") {
            const mathContent = trimmed.slice(2, -2);
            result.push("$$");
            result.push(mathContent);
            result.push("$$");
        } else {
            result.push(line);
        }

        prevWasTable = isTable;
    }

    return result.join("\n");
}

function extractText(node: ReactNode): string {
    if (typeof node === "string") {
        return node;
    }

    if (typeof node === "number") {
        return String(node);
    }

    if (!node) {
        return "";
    }

    if (Array.isArray(node)) {
        return node.map(extractText).join("");
    }

    if (isValidElement(node)) {
        const props = node.props as { children?: ReactNode };
        return extractText(props.children);
    }

    return "";
}

function CodeBlock({ language, children }: { language: string; children: ReactNode }) {
    const [copied, setCopied] = useState(false);
    const theme = useThemeStore((state) => state.theme);
    const isDark = theme === "dark";
    const code = extractText(children).replace(/\n$/, "");

    const handleCopy = () => {
        navigator.clipboard.writeText(code).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    return (
        <div className="group relative my-2">
            {language && <span className="absolute top-2 right-2 text-[9px] uppercase text-muted-foreground/40 font-mono select-none z-10 pointer-events-none">{language}</span>}
            <button
                onClick={handleCopy}
                className={cn(
                    "absolute top-2 left-2 p-1 rounded-md text-muted-foreground/50 hover:text-muted-foreground transition-all opacity-0 group-hover:opacity-100 z-10",
                    isDark ? "bg-white/5 hover:bg-white/10" : "bg-black/5 hover:bg-black/10",
                )}
                title="Sao chep code"
            >
                {copied ? <ClipboardCheck className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
            </button>
            <SyntaxHighlighter
                language={language}
                style={isDark ? oneDark : oneLight}
                PreTag="div"
                customStyle={{
                    margin: 0,
                    borderRadius: "8px",
                    fontSize: "12px",
                    padding: "10px 12px",
                    ...(isDark
                        ? {
                              background: "oklch(0.18 0.015 155)",
                              border: "1px solid oklch(0.30 0.025 155)",
                          }
                        : {
                              background: "oklch(0.96 0.008 105)",
                              border: "1px solid oklch(0.88 0.018 105)",
                          }),
                }}
                codeTagProps={{ style: { fontFamily: '"IBM Plex Mono", "Fira Code", monospace' } }}
            >
                {code}
            </SyntaxHighlighter>
        </div>
    );
}

interface MarkdownWithCitationsProps {
    content: string;
    sources: ChatSourceChunk[];
    relatedEntities: string[];
    imageRefs?: ChatImageRef[];
}

export function MarkdownWithCitations({ content, sources, relatedEntities, imageRefs }: MarkdownWithCitationsProps) {
    const processed = preprocessMarkdown(content);
    const allSources = useContext(AllSourcesCtx);

    const withCitations = (Tag: string) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return ({ children, ...props }: any) => {
            const injected = injectCitations(children, sources, relatedEntities, imageRefs, allSources);
            return <Tag {...props}>{injected}</Tag>;
        };
    };

    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
                p: withCitations("p"),
                li: withCitations("li"),
                td: withCitations("td"),
                th: withCitations("th"),
                h1: withCitations("h1"),
                h2: withCitations("h2"),
                h3: withCitations("h3"),
                h4: withCitations("h4"),
                h5: withCitations("h5"),
                h6: withCitations("h6"),
                strong: withCitations("strong"),
                em: withCitations("em"),
                a: ({ href, children, ...props }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                        {injectCitations(children, sources, relatedEntities, imageRefs, allSources)}
                    </a>
                ),
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                code: ({ className, children, ...props }: any) => {
                    const langMatch = /language-(\w+)/.exec(className || "");
                    if (!langMatch) {
                        return (
                            <code className={className} {...props}>
                                {children}
                            </code>
                        );
                    }

                    return <CodeBlock language={langMatch[1]}>{children}</CodeBlock>;
                },
            }}
        >
            {processed}
        </ReactMarkdown>
    );
}
