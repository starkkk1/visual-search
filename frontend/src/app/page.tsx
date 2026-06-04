"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, Loader2, Image as ImageIcon } from "lucide-react";

interface SearchResult {
  path: string;
  url: string;
  score: number;
}

export default function Home() {
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/search", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Search failed");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError("An error occurred while searching. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: false,
  });

  return (
    <main className="max-w-6xl mx-auto px-4 py-16 sm:px-6 lg:px-8 flex flex-col items-center">
      <header className="text-center mb-16 animate-fade-in-down">
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-br from-white to-zinc-400 drop-shadow-sm">
          Visual Search
        </h1>
        <p className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto">
          Find similar images instantly using deep learning.
        </p>
      </header>

      <div
        {...getRootProps()}
        className={`w-full max-w-3xl rounded-3xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300 backdrop-blur-xl animate-fade-in-up flex flex-col items-center justify-center gap-4 ${
          isDragActive
            ? "border-purple-500 bg-purple-500/10 shadow-[0_10px_40px_-10px_rgba(139,92,246,0.3)] scale-[1.02]"
            : "border-zinc-700 bg-zinc-900/50 hover:border-purple-500 hover:bg-purple-500/5 hover:shadow-[0_10px_40px_-10px_rgba(139,92,246,0.15)] hover:-translate-y-1"
        }`}
        style={{ animationDelay: "0.2s" }}
      >
        <input {...getInputProps()} />
        <div className="p-4 rounded-full bg-purple-500/20 text-purple-400 mb-2">
          <UploadCloud className="w-10 h-10" />
        </div>
        <div className="space-y-1">
          <p className="text-2xl font-semibold text-zinc-100">
            {isDragActive ? "Drop the image here" : "Drag & drop an image here"}
          </p>
          <p className="text-zinc-400">or click to browse from your device</p>
        </div>
      </div>

      {error && (
        <div className="mt-8 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 animate-fade-in-up">
          {error}
        </div>
      )}

      {loading && (
        <div className="mt-16 flex flex-col items-center gap-4 animate-fade-in-up">
          <Loader2 className="w-10 h-10 text-purple-500 animate-spin" />
          <p className="text-zinc-400 font-medium tracking-wide">Analyzing visual features...</p>
        </div>
      )}

      {results && (
        <div className="mt-16 w-full animate-fade-in-up">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-zinc-100">Top Matches</h2>
            <span className="px-3 py-1 text-sm font-medium rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/20">
              {results.length} results
            </span>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {results.map((result, idx) => (
              <ResultCard key={result.path} result={result} index={idx} />
            ))}
          </div>
        </div>
      )}
    </main>
  );
}

function ResultCard({ result, index }: { result: SearchResult; index: number }) {
  const percentage = (result.score * 100).toFixed(1);
  const filename = result.path.split(/[/\\]/).pop() || "";

  return (
    <div
      className="group relative rounded-2xl overflow-hidden bg-zinc-900/80 border border-zinc-800 backdrop-blur-md transition-all duration-500 hover:-translate-y-2 hover:border-purple-500/50 hover:shadow-[0_20px_40px_-15px_rgba(139,92,246,0.3)] opacity-0 animate-[fadeInUp_0.5s_ease-out_forwards]"
      style={{ animationDelay: `${0.1 + index * 0.05}s` }}
    >
      <div className="aspect-square overflow-hidden bg-zinc-950 relative">
        <img
          src={result.url}
          alt={filename}
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.onerror = null;
            target.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%' viewBox='0 0 100 100'><rect width='100%' height='100%' fill='%2309090b'/><text x='50%' y='50%' fill='%2352525b' font-family='sans-serif' font-size='10' text-anchor='middle' dominant-baseline='middle'>Image Missing</text></svg>";
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-zinc-950/90 via-zinc-950/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 overflow-hidden">
            <ImageIcon className="w-4 h-4 text-zinc-400 shrink-0" />
            <span className="text-sm font-medium text-zinc-300 truncate" title={filename}>
              {filename}
            </span>
          </div>
          <span className="shrink-0 inline-flex items-center justify-center px-2 py-1 rounded-md bg-purple-500/20 text-purple-300 text-xs font-bold ring-1 ring-inset ring-purple-500/30">
            {percentage}%
          </span>
        </div>
      </div>
    </div>
  );
}
