import os
import re


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.indexes = []


class SubstringTrie:
    def __init__(self, k: int = 3):
        self.root = TrieNode()
        self.k = k

    def insert(self, substring: str, exact_index: int) -> None:

        node = self.root
        for char in substring:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True

        if not node.indexes or node.indexes[-1] != exact_index:
            node.indexes.append(exact_index)

    def extract_and_insert_substrings(self, word: str, exact_index: int) -> None:

        n = len(word)

        if n < self.k:
            self.insert(word, exact_index)
            return

        for i in range(n - self.k + 1):
            sub = word[i: i + self.k]
            self.insert(sub, exact_index)

    def build_from_text(self, full_text: str) -> None:

        for match in re.finditer(r'\S+', full_text):
            word = match.group()


            exact_index = match.start()


            clean_word = word.lower().strip(".,!?()[]{}\"'")

            if clean_word:
                self.extract_and_insert_substrings(clean_word, exact_index)

    def get_all_substrings(self) -> list[tuple[str, list[int]]]:

        results = []

        def _dfs(node: TrieNode, current_prefix: str):
            if node.is_end_of_word:
                results.append((current_prefix, node.indexes))

            for char, child_node in node.children.items():
                _dfs(child_node, current_prefix + char)

        _dfs(self.root, "")
        return results



if __name__ == "__main__":
    file_path = input("Enter file path: ").strip().strip("'\"")

    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
    else:
        try:

            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            trie = SubstringTrie(k=3)

            trie.build_from_text(full_text)

            all_substrings = trie.get_all_substrings()

            print(f"\n--- Substrings (Total: {len(all_substrings)}) ---")
            for sub, idxs in all_substrings:
                print(f"Substring: '{sub}' ➔ Exact File Indexes: {idxs}")

        except Exception as e:
            print(f" Error reading file: {e}")