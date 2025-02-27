import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from tqdm import tqdm
import itertools
import numpy as np
from scipy.stats import norm
from scipy.sparse import lil_matrix
import random
from sklearn.cluster import KMeans
import math
from collections import defaultdict

class Battleground:
    def __init__(self, csv_path, openai_api_key=None, min_comparisons_per_person=5):
        """Initialize the battleground with a CSV of people.
        
        Args:
            csv_path: Path to CSV containing people names in a 'name' column
            openai_api_key: OpenAI API key
            min_comparisons_per_person: Minimum number of comparisons each person should get
        """
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError("OpenAI API key must be provided or set in environment variables")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.2,
            openai_api_key=openai_api_key
        )
        
        # Read CSV and limit to 2500 rows
        self.people_df = pd.read_csv(csv_path)
        if 'name' not in self.people_df.columns:
            raise ValueError("CSV must contain a 'name' column")
        self.people_df = self.people_df.head(500)  # Limit to 2500 rows
        self.people = list(self.people_df['name'])
        self.n_people = len(self.people)
        self.min_comparisons = min_comparisons_per_person
        
        # Calculate parameters based on dataset size
        self.n_clusters = self._calculate_clusters()
        self.samples_per_cluster = self._calculate_samples_per_cluster()
        self.cross_cluster_samples = min(300, self.n_people)
        
        # Use sparse matrices for large datasets
        self.comparison_matrix = lil_matrix((self.n_people, self.n_people))
        self.comparison_counts = lil_matrix((self.n_people, self.n_people))
        self.comparison_history = defaultdict(list)  # Track comparison history per person
        self.total_comparisons = 0  # Add counter for total comparisons
        
        self.comparison_prompt = PromptTemplate(
            input_variables=["person1", "person2"],
            template="""Between {person1} and {person2}, who would you consider more important or influential? 
            Consider their achievements, impact on society, historical significance, and lasting legacy.
            
            Respond with ONLY the name of the person you consider more important. Just the name, nothing else."""
        )

    def _calculate_clusters(self):
        """Calculate optimal number of clusters based on dataset size."""
        if self.n_people < 100:
            return max(5, self.n_people // 10)
        elif self.n_people < 1000:
            return int(math.sqrt(self.n_people))
        else:
            return min(100, int(math.sqrt(self.n_people) * math.log10(self.n_people)))

    def _calculate_samples_per_cluster(self):
        """Calculate samples per cluster based on dataset size."""
        if self.n_people < 100:
            return min(20, self.n_people // 2)
        elif self.n_people < 1000:
            return min(50, self.n_people // 10)
        else:
            return min(100, self.n_people // 20)

    def compare_pair(self, person1_idx, person2_idx):
        """Compare two people using the LLM and return the winner index."""
        person1 = self.people[person1_idx]
        person2 = self.people[person2_idx]
        
        prompt = self.comparison_prompt.format(person1=person1, person2=person2)
        response = self.llm.invoke(prompt).content.strip()
        
        winner = None
        if person1.lower() in response.lower():
            self.comparison_matrix[person1_idx, person2_idx] += 1
            winner = person1_idx
        elif person2.lower() in response.lower():
            self.comparison_matrix[person2_idx, person1_idx] += 1
            winner = person2_idx
            
        if winner is not None:
            self.comparison_counts[person1_idx, person2_idx] += 1
            self.comparison_counts[person2_idx, person1_idx] += 1
            self.comparison_history[person1_idx].append((person2_idx, winner == person1_idx))
            self.comparison_history[person2_idx].append((person1_idx, winner == person2_idx))
            self.total_comparisons += 1
            
            # Print progress every 100 comparisons
            if self.total_comparisons % 100 == 0:
                print(f"\nComparisons completed: {self.total_comparisons}")
                print(f"Latest comparison: {person1} vs {person2} -> Winner: {self.people[winner]}")
                current_scores = self.get_current_scores()
                top_3_idx = np.argsort(-current_scores)[:3]
                print("Current top 3:")
                for idx in top_3_idx:
                    print(f"  {self.people[idx]}: {current_scores[idx]:.3f}")
            
        return winner

    def get_current_scores(self):
        """Get current scores based on comparison matrix with win ratio."""
        total_comparisons = np.array(self.comparison_counts.sum(axis=1)).flatten()
        wins = np.array(self.comparison_matrix.sum(axis=1)).flatten()
        # Use win ratio but handle division by zero
        scores = np.divide(wins, total_comparisons, out=np.zeros_like(wins), where=total_comparisons!=0)
        return scores

    def get_undersampled_people(self):
        """Get indices of people with fewer than minimum comparisons."""
        comparisons = np.array(self.comparison_counts.sum(axis=1)).flatten()
        return np.where(comparisons < self.min_comparisons)[0]

    def cluster_people(self):
        """Cluster people based on current scores and comparison counts."""
        scores = self.get_current_scores()
        comparisons = np.array(self.comparison_counts.sum(axis=1)).flatten()
        
        # Combine scores and comparison counts for clustering
        features = np.column_stack([
            scores,
            comparisons / max(1, comparisons.max())  # Normalize comparison counts
        ])
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        return kmeans.fit_predict(features)

    def run_tournament(self):
        """Run tournament using adaptive sampling approach."""
        print(f"Dataset size: {self.n_people} people")
        print(f"Using {self.n_clusters} clusters with {self.samples_per_cluster} samples per cluster")
        print(f"Minimum comparisons per person: {self.min_comparisons}")
        
        # Phase 1: Initial random sampling within clusters
        print("Phase 1: Initial cluster sampling...")
        clusters = self.cluster_people()
        for cluster_id in tqdm(range(self.n_clusters)):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) < 2:
                continue
            
            pairs = list(itertools.combinations(cluster_indices, 2))
            random.shuffle(pairs)
            for i, j in pairs[:self.samples_per_cluster]:
                self.compare_pair(i, j)
        
        # Phase 2: Cross-cluster sampling with focus on promising candidates
        print("Phase 2: Cross-cluster sampling...")
        scores = self.get_current_scores()
        
        # Get top performers from each cluster
        top_per_cluster = []
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            # Get top 2 from each cluster to increase diversity
            cluster_scores = scores[cluster_indices]
            top_2_idx = cluster_indices[np.argsort(-cluster_scores)[:2]]
            top_per_cluster.extend(top_2_idx)
        
        # Compare top performers across clusters
        cross_pairs = list(itertools.combinations(top_per_cluster, 2))
        random.shuffle(cross_pairs)
        for i, j in tqdm(cross_pairs[:self.cross_cluster_samples]):
            self.compare_pair(i, j)
        
        # Phase 3: Ensure minimum comparisons for all people
        print("Phase 3: Ensuring minimum comparisons...")
        undersampled = self.get_undersampled_people()
        while len(undersampled) > 0:
            for idx in tqdm(undersampled):
                # Compare with mix of random and nearby-ranked people
                current_scores = self.get_current_scores()
                rank_diff = np.abs(current_scores - current_scores[idx])
                potential_opponents = np.argsort(rank_diff)
                
                # Remove already heavily compared pairs
                potential_opponents = [
                    opp for opp in potential_opponents 
                    if opp != idx and self.comparison_counts[idx, opp] < 3
                ]
                
                if potential_opponents:
                    # Mix of close and random opponents
                    if random.random() < 0.7:  # 70% chance of close opponent
                        opponent = potential_opponents[0]
                    else:
                        opponent = random.choice(potential_opponents)
                    self.compare_pair(idx, opponent)
            
            undersampled = self.get_undersampled_people()
        
        # Phase 4: Final refinement of top contestants
        print("Phase 4: Refining top rankings...")
        scores = self.get_current_scores()
        top_k = min(50, self.n_people // 10)
        top_indices = np.argsort(-scores)[:top_k]
        top_pairs = list(itertools.combinations(top_indices, 2))
        random.shuffle(top_pairs)
        
        for i, j in tqdm(top_pairs[:100]):
            self.compare_pair(i, j)

    def get_rankings(self):
        """Get the final rankings with detailed statistics."""
        scores = self.get_current_scores()
        total_comparisons = np.array(self.comparison_counts.sum(axis=1)).flatten()
        wins = np.array(self.comparison_matrix.sum(axis=1)).flatten()
        
        rankings = pd.DataFrame({
            'Name': self.people,
            'Win_Ratio': scores,
            'Wins': wins,
            'Total_Comparisons': total_comparisons,
            'Average_Opponent_Rank': [
                np.mean([scores[opp] for opp, _ in self.comparison_history[i]])
                if self.comparison_history[i] else 0
                for i in range(self.n_people)
            ]
        })
        
        rankings = rankings.sort_values('Win_Ratio', ascending=False).reset_index(drop=True)
        rankings.index = rankings.index + 1  # Start ranking from 1
        return rankings

    def save_rankings(self, output_path):
        """Save the rankings to a CSV file with detailed statistics."""
        rankings = self.get_rankings()
        rankings.to_csv(output_path, index_label='Rank')
        total_comparisons = int(self.comparison_counts.sum() / 2)
        print(f"\nStatistics:")
        print(f"Total comparisons made: {total_comparisons}")
        print(f"Average comparisons per person: {total_comparisons/self.n_people:.1f}")
        print(f"Rankings saved to {output_path}")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    input_csv = "person_2020_update.csv"
    output_csv = "rankings.csv"
    
    battleground = Battleground(input_csv, min_comparisons_per_person=10)
    battleground.run_tournament()
    battleground.save_rankings(output_csv)

if __name__ == "__main__":
    main()
