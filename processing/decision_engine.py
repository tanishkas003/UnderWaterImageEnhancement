import numpy as np

class DecisionEngine:
    def __init__(self):
        pass

    # -----------------------------
    # Rule-based technique selection
    # -----------------------------
    def select_techniques(self, features):
        techniques = []

        r = features['r']
        g = features['g']
        b = features['b']
        brightness = features['brightness']
        contrast = features['contrast']
        entropy = features['entropy']

        # Rule 1: Strong blue dominance → color correction needed
        if b > r + 30:
            techniques.append("white_balance")

        # Rule 2: Low brightness → gamma correction
        if brightness < 80:
            techniques.append("gamma")

        # Rule 3: Low contrast → CLAHE
        if contrast < 40:
            techniques.append("clahe")

        # Rule 4: Low entropy → sharpening
        if entropy < 5:
            techniques.append("sharpen")

        # Default fallback
        if len(techniques) == 0:
            techniques.append("fusion")

        return techniques

    # -----------------------------
    # BONUS 2: Score-based ranking
    # -----------------------------
    def rank_techniques(self, features):
        scores = {}

        brightness = features['brightness']
        contrast = features['contrast']
        entropy = features['entropy']
        r = features['r']
        b = features['b']

        # Higher score = more needed

        # CLAHE → improves contrast
        scores['clahe'] = contrast

        # Sharpen → improves detail
        scores['sharpen'] = entropy

        # Gamma → helps dark images
        scores['gamma'] = 255 - brightness

        # White balance → corrects blue dominance
        scores['white_balance'] = max(0, b - r)

        # Fusion → general enhancement baseline
        scores['fusion'] = 50  # neutral score

        # Sort descending (highest priority first)
        ranked = sorted(scores, key=scores.get, reverse=True)

        return ranked, scores

    # -----------------------------
    # FINAL selector (combined logic)
    # -----------------------------
    def get_final_techniques(self, features, top_k=2):
        """
        Combines:
        - Rule-based filtering
        - Score-based ranking
        - Multi-technique selection (top_k)
        """

        # Step 1: rule-based shortlist
        selected = self.select_techniques(features)

        # Step 2: ranking
        ranked, scores = self.rank_techniques(features)

        # Step 3: keep only relevant ones from ranked
        filtered_ranked = [t for t in ranked if t in selected]

        # Step 4: pick top_k techniques
        final = filtered_ranked[:top_k]

        # Safety fallback
        if len(final) == 0:
            final = ["fusion"]

        return final, scores