from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []
        for i in range(num_merges):
            if len(tokens) < 2:
                break
            pairs = {}
            for j in range(len(tokens) - 1):
                p = (tokens[j], tokens[j+1])
                pairs[p] = 1 + pairs.get(p, 0)
            if not pairs:
                break
            
            best_cnt = max(pairs.values())
            c = sorted(p for p,c in pairs.items() if c == best_cnt)
            best = c[0]
            merges.append([best[0], best[1]])

            new = []
            x = 0
            while x < len(tokens):
                if x < len(tokens) - 1 and tokens[x] == best[0] and tokens[x+1] == best[1]:
                    new.append(best[0] + best[1])
                    x += 2
                else:
                    new.append(tokens[x])
                    x += 1
            tokens = new
        return merges
