#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>

// Define the constant 2^k for age granularity
const uint32_t k = 2;

// Function to compute EVA and update ranks
std::vector<double> computeEVAandUpdateRanks(std::vector<std::vector<double>>& hitCtrs, std::vector<std::vector<double>>& evictionCtrs, double A, double N) {
    // Initialize variables
    std::cout << "Ranking1:\n";
    std::vector<double> rank(2 * (1 << (k + 1)));
    std::cout << "Ranking2:\n";
    std::vector<double> hR(2 * (1 << (k + 1)));
    std::cout << "Ranking3:\n";
    std::vector<double> hNR(2 * (1 << (k + 1)));
    std::cout << "Ranking4:\n";
    std::vector<double> eventR(2 * (1 << (k + 1)));
    std::cout << "Ranking3:\n";
    std::vector<double> eventNR(2 * (1 << (k + 1)));
    // Compute hit rates from counters
    for (double a = (1 << k); a >= 1; a--) {
        double hitsR = 0, hitsNR = 0;
        double eventsR = 0, eventsNR = 0;
        for (double c = 0; c < 2; c++) { // 0 - NotReused; 1 - Reused
            if (c == 0) {
                hitsNR += hitCtrs[c][a];
                eventsNR += hitCtrs[c][a] + evictionCtrs[c][a];
            }
            else {
                hitsR += hitCtrs[c][a];
                eventsR += hitCtrs[c][a] + evictionCtrs[c][a];
            }
        }
        double c = (hitsR / eventsR);
        hR[a] = c;
        hNR[a] = (hitsNR) / eventsNR;
        eventR[a] = eventsR;
        eventNR[a] = eventsNR;
    }

    //double h = (hR + hNR) / 2;
    double h = 0;
    for (double a = (1 << k); a >= 1; a--) {
        h += (hR[a] + hNR[a]);
    }
    double events = 0;
    for (double a = (1 << k); a >= 1; a--) {
        events += (eventR[a] + eventNR[a]);
    }
    h = h / events;
    std::cout << "Value of (hitsR) / eventsR: " << h << std::endl;
    double perAccessCost = h * A / N;
    std::cout << "Value of (hitsR) / eventsR: " << perAccessCost << std::endl;
    // Compute EVA
    std::vector<std::vector<double>> eva(2, std::vector<double>(2 * (1 << (k + 1)), 0));
    double evaReused = eva[1][1] / (1 - hR[0]);

    for (double c = 0; c < 2; c++) {
        double expectedLifetime = 0;
        double hits = 0, events = 0;
        for (double a = (1 << k); a >= 1; a--) {
            expectedLifetime += events;
            eva[c][a] = (hits - perAccessCost * expectedLifetime) / events;
            hits += hitCtrs[c][a];
            events += hitCtrs[c][a] + evictionCtrs[c][a];
        }
    }

    // Differentiate classes
    for (double c = 0; c < 2; c++) {
        for (double a = (1 << k); a >= 1; a--) {
            eva[c][a] += (hR[a] - h) * evaReused;
        }
    }

    // Finally, rank ages by EVA
    std::vector<double> evaWithIndex(2 * (1 << (k + 1)));
    for (double i = 0; i <= (2 * (1 << (k + 1)))-1; i++) {
        evaWithIndex[i] = i;
    }

    std::sort(evaWithIndex.begin(), evaWithIndex.end(), [&](double a, double b) {
        return eva[0][a] > eva[0][b];
        });

    for (double i = 0; i < (2 * (1 << (k + 1))); i++) {
        rank[evaWithIndex[i]] = (2 * (1 << (k + 1))) - i;
    }
    std::cout << "Rankingrank:\n";
    return rank;
}

int main() {
    // Example usage
    std::vector<std::vector<double>> hitCtrs = { {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8} };
    std::vector<std::vector<double>> evictionCtrs = { {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8} };
    double A = 1, N = 8;
    std::cout << "Ranking:\n";
    std::vector<double> rank = computeEVAandUpdateRanks(hitCtrs, evictionCtrs, A, N);

    std::cout << "Ranking:\n";
    for (double i = 0; i <= (2 * (1 << (k + 1)))-1; i++) {
        std::cout << "Age " << i << ": " << rank[i] << std::endl;
    }

    return 0;
}