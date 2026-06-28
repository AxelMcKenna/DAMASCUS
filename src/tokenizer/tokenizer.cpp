#include "tokenizer.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace damascus {

namespace {

// ---- UTF-8 ---------------------------------------------------------------

// Decode one codepoint at byte offset `i`. Returns the codepoint and advances
// `i` past it. Malformed bytes are passed through as Latin-1 (1 byte), which
// keeps the tokenizer total — byte-level BPE can represent any byte anyway.
uint32_t utf8_next(const std::string& s, std::size_t& i) {
    auto b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) { i += 1; return b; }
    if ((b >> 5) == 0x6 && i + 1 < s.size()) {
        uint32_t cp = (b & 0x1F) << 6 | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        i += 2; return cp;
    }
    if ((b >> 4) == 0xE && i + 2 < s.size()) {
        uint32_t cp = (b & 0x0F) << 12
                    | (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6
                    | (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        i += 3; return cp;
    }
    if ((b >> 3) == 0x1E && i + 3 < s.size()) {
        uint32_t cp = (b & 0x07) << 18
                    | (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12
                    | (static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6
                    | (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        i += 4; return cp;
    }
    i += 1; return b;
}

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// ---- codepoint classes for the pre-tokenizer regex -----------------------
//
// The llama-3 pre-tokenizer regex uses Unicode properties (\p{L}, \p{N}, \s).
// We classify ASCII exactly and approximate non-ASCII as "letter" (the common
// case for natural-language text). Full ICU-grade property tables are a known
// follow-up; they do not affect ASCII prompts such as the Milestone-A oracle.

bool is_newline(uint32_t c) { return c == '\r' || c == '\n'; }

bool is_whitespace(uint32_t c) {
    switch (c) {
        case ' ': case '\t': case '\n': case '\r': case '\v': case '\f':
        case 0x85: case 0xA0: case 0x2028: case 0x2029:
            return true;
        default:
            return (c >= 0x2000 && c <= 0x200A);
    }
}

bool is_number(uint32_t c) { return c >= '0' && c <= '9'; }

bool is_letter(uint32_t c) {
    if (c < 0x80) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    // Non-ASCII: treat as letter unless it is whitespace.
    return !is_whitespace(c);
}

}  // namespace

Tokenizer::Tokenizer(const TokenizerData& data) {
    if (data.model != "gpt2") {
        throw std::runtime_error(
            "Error: unsupported tokenizer model (expected gpt2): " + data.model);
    }

    // --- vocab ---
    id_to_token_ = data.tokens;
    token_to_id_.reserve(data.tokens.size());
    for (int32_t id = 0; id < static_cast<int32_t>(data.tokens.size()); ++id) {
        token_to_id_.emplace(data.tokens[id], id);
    }

    // --- merge ranks ---
    merge_rank_.reserve(data.merges.size());
    for (int32_t rank = 0; rank < static_cast<int32_t>(data.merges.size()); ++rank) {
        const std::string& m = data.merges[rank];
        // "A B": A and B never contain a literal space (space -> 'Ġ' byte-char),
        // so splitting on the first space is unambiguous.
        std::size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank_.emplace(m, rank);  // key matches "left + ' ' + right"
    }

    bos_id_ = data.bos_token_id;
    eos_id_ = data.eos_token_id;

    // --- GPT-2 byte <-> unicode table ---
    // Printable byte ranges map to themselves; the rest map to U+0100.. in
    // order of first appearance, so every byte becomes a visible codepoint.
    std::array<bool, 256> printable{};
    auto mark = [&](int lo, int hi) { for (int b = lo; b <= hi; ++b) printable[b] = true; };
    mark('!', '~');      // 0x21..0x7E
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    int next = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = printable[b] ? static_cast<uint32_t>(b)
                                   : static_cast<uint32_t>(256 + next++);
        std::string ch = utf8_encode(cp);
        byte_to_unicode_[b] = ch;
        unicode_to_byte_.emplace(std::move(ch), static_cast<uint8_t>(b));
    }
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    // Decode the whole string to codepoints once, tracking byte offsets so each
    // chunk can be sliced back out as raw bytes for the byte-level encoder.
    std::vector<uint32_t> cp;
    std::vector<std::size_t> off;  // off[k] = byte offset of cp[k]; off[n] = size
    for (std::size_t i = 0; i < text.size();) {
        off.push_back(i);
        cp.push_back(utf8_next(text, i));
    }
    off.push_back(text.size());
    const std::size_t n = cp.size();

    std::vector<std::string> chunks;
    std::size_t i = 0;

    auto emit = [&](std::size_t start, std::size_t end) {
        chunks.push_back(text.substr(off[start], off[end] - off[start]));
    };

    while (i < n) {
        std::size_t start = i;

        // alt1: contractions  (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[i] == '\'' && i + 1 < n) {
            auto lower = [&](std::size_t k) {
                uint32_t c = cp[k];
                return (c >= 'A' && c <= 'Z') ? c + 32 : c;
            };
            uint32_t a = lower(i + 1);
            uint32_t b = (i + 2 < n) ? lower(i + 2) : 0;
            std::size_t len = 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') len = 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
                     (a == 'l' && b == 'l')) len = 3;
            if (len) { i += len; emit(start, i); continue; }
        }

        // alt2:  [^\r\n\p{L}\p{N}]? \p{L}+
        {
            std::size_t j = i;
            // optional single prefix char, only if a letter follows it
            if (!is_newline(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j]) &&
                j + 1 < n && is_letter(cp[j + 1])) {
                j += 1;
            }
            if (j < n && is_letter(cp[j])) {
                while (j < n && is_letter(cp[j])) ++j;
                i = j; emit(start, i); continue;
            }
        }

        // alt3:  \p{N}{1,3}
        if (is_number(cp[i])) {
            std::size_t j = i;
            while (j < n && is_number(cp[j]) && (j - i) < 3) ++j;
            i = j; emit(start, i); continue;
        }

        // alt4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            auto is_other = [&](uint32_t c) {
                return !is_whitespace(c) && !is_letter(c) && !is_number(c);
            };
            std::size_t j = i;
            if (cp[j] == ' ' && j + 1 < n && is_other(cp[j + 1])) ++j;  // optional space
            if (j < n && is_other(cp[j])) {
                while (j < n && is_other(cp[j])) ++j;
                while (j < n && is_newline(cp[j])) ++j;
                i = j; emit(start, i); continue;
            }
        }

        // alt5:  \s*[\r\n]+   (whitespace run that ends in newline(s))
        if (is_whitespace(cp[i])) {
            std::size_t j = i, last_nl = std::string::npos;
            while (j < n && is_whitespace(cp[j])) {
                if (is_newline(cp[j])) last_nl = j;
                ++j;
            }
            if (last_nl != std::string::npos) {
                i = last_nl + 1; emit(start, i); continue;
            }
        }

        // alt6:  \s+(?!\S)   /  alt7: \s+
        if (is_whitespace(cp[i])) {
            std::size_t j = i;
            while (j < n && is_whitespace(cp[j])) ++j;
            // (?!\S): if a non-whitespace follows, leave its leading space behind
            if (j < n && (j - i) > 1) i = j - 1;
            else i = j;
            emit(start, i); continue;
        }

        // No alternative matched (shouldn't happen) — consume one codepoint.
        i += 1; emit(start, i);
    }

    return chunks;
}

std::vector<std::string> Tokenizer::bpe(const std::string& chunk) const {
    // Map raw bytes to byte-chars; each becomes one BPE symbol.
    std::vector<std::string> symbols;
    symbols.reserve(chunk.size());
    for (unsigned char b : chunk) symbols.push_back(byte_to_unicode_[b]);
    if (symbols.size() < 2) return symbols;

    // Greedily apply the lowest-rank adjacent merge until none apply.
    while (true) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::size_t best_i = 0;
        bool found = false;

        for (std::size_t k = 0; k + 1 < symbols.size(); ++k) {
            std::string key = symbols[k] + ' ' + symbols[k + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = k;
                found = true;
            }
        }
        if (!found) break;

        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
    }
    return symbols;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos) ids.push_back(static_cast<int32_t>(bos_id_));

    for (const std::string& chunk : pretokenize(text)) {
        for (const std::string& piece : bpe(chunk)) {
            auto it = token_to_id_.find(piece);
            if (it == token_to_id_.end()) {
                throw std::runtime_error(
                    "Error: BPE produced piece absent from vocab: " + piece);
            }
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Concatenate the byte-char pieces, then map each codepoint back to its
    // original byte. The result is the raw UTF-8 text.
    std::string unicode;
    for (int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) {
            throw std::runtime_error("Error: token id out of range in decode");
        }
        unicode += id_to_token_[static_cast<std::size_t>(id)];
    }

    std::string bytes;
    for (std::size_t i = 0; i < unicode.size();) {
        std::size_t start = i;
        utf8_next(unicode, i);
        std::string ch = unicode.substr(start, i - start);
        auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            bytes.push_back(static_cast<char>(it->second));
        } else {
            // Non-byte-level piece (e.g. a control/special token added verbatim);
            // emit it literally.
            bytes += ch;
        }
    }
    return bytes;
}

}  // namespace damascus
