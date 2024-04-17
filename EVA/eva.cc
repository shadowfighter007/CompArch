#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
#include <iostream>
#include <random>
#include <cstdint>
#include <bitset>
#include <tuple>
#include "cache.h"

namespace
{
const uint32_t k = 14; // Define the constant 2^k for age granularity
const uint32_t size = 2 * (1 << (k + 1));
const int NUM_SETS = 2048; // Number of sets in the cache
const int NUM_WAYS = 16; // Number of ways (cache lines) per set
const int AGE_BITS = 14; // Number of bits for age counter
const int TIMESTAMP_BITS = 7; // Number of bits for timestamp counter
const int AGE_GRANULARITY = 15; // Age granularity (limit A)
const int MAX_AGE_VALUE = 2 * (1 << (k + 1)); // Maximum value for age counter
std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;
double hitCtrs_nn[MAX_AGE_VALUE][2] = {{0.0}};
double evictionCtrs_nn[MAX_AGE_VALUE][2] = {{0.0}};
int evafg = 0;
double ewmaDecay = 0.8;
}

// Cache line structure
struct CacheLine {
    std::bitset<AGE_BITS> ageCounter;
    bool classificationBit;

    CacheLine() : ageCounter(0), classificationBit(0) {}
};

// Cache set structure
struct CacheSet {
    std::vector<CacheLine> lines;
    std::bitset<TIMESTAMP_BITS> timestamp;

    CacheSet() : lines(NUM_WAYS), timestamp(0) {}
};

// Cache structure
class Cache_Age {
private:
    std::vector<CacheSet> sets;

public:
    Cache_Age() : sets(NUM_SETS) {}

    void accessCache(int set, int way, bool hit, bool evict) {
        CacheSet& currentSet = sets[set];
        currentSet.timestamp = (currentSet.timestamp.to_ulong() + 1) % AGE_GRANULARITY;

        for (auto& line : currentSet.lines) {
            if (line.ageCounter.to_ulong() < MAX_AGE_VALUE) {
                line.ageCounter = line.ageCounter.to_ulong() + 1;
            }
        }

        if (hit) {
            currentSet.lines[way].ageCounter.reset();
            currentSet.lines[way].classificationBit = 1;
        }
        if (evict) {
            currentSet.lines[way].ageCounter.reset();
            currentSet.lines[way].classificationBit = 0;
        }
    }
    std::tuple<unsigned int, bool, unsigned int> getCacheLineInfo(int set, int way) {
    CacheSet& currentSet = sets[set];
    CacheLine& line = currentSet.lines[way];

    return std::make_tuple(line.ageCounter.to_ulong(), line.classificationBit, currentSet.timestamp.to_ulong());
    }

    void printCacheStatus(int set, int way) {
        CacheSet& currentSet = sets[set];
        CacheLine& line = currentSet.lines[way];

        std::cout << "Set: " << set << ", Way: " << way << std::endl;
        std::cout << "Age Counter: " << line.ageCounter << std::endl;
        std::cout << "Classification Bit: " << line.classificationBit << std::endl;
        std::cout << "Timestamp: " << currentSet.timestamp << std::endl;
    }
    
    void printCacheStatus(int set) {
        CacheSet& currentSet = sets[set];
        std::cout << "Set: " << set << std::endl;
            for (int way = 0; way < NUM_WAYS; way++) {
                CacheLine& line = currentSet.lines[way];
                std::cout << "Way: " << way << ", Age Counter: " << line.ageCounter.to_ulong()
                          << ", Classification Bit: " << line.classificationBit << std::endl;
            }
            std::cout << "Timestamp: " << currentSet.timestamp.to_ulong() << std::endl;
            std::cout << std::endl;
    }

    void printCacheStatus() {
        for (int set = 0; set < NUM_SETS; set++) {
            CacheSet& currentSet = sets[set];

            std::cout << "Set: " << set << std::endl;
            for (int way = 0; way < NUM_WAYS; way++) {
                CacheLine& line = currentSet.lines[way];
                std::cout << "Way: " << way << ", Age Counter: " << line.ageCounter.to_ulong()
                          << ", Classification Bit: " << line.classificationBit << std::endl;
            }
            std::cout << "Timestamp: " << currentSet.timestamp.to_ulong() << std::endl;
            std::cout << std::endl;
        }
    }
};

// Function to compute EVA and update ranks
std::vector<double> computeEVAandUpdateRanks(double hitCtrs[MAX_AGE_VALUE][2], double evictionCtrs[MAX_AGE_VALUE][2], double A, double N) {
    // Initialize variables
    std::vector<double> rank(2 * (1 << (k + 1)));
    std::vector<double> hR(2 * (1 << (k + 1)));
    std::vector<double> hNR(2 * (1 << (k + 1)));
    double hRsum = 0;
    double hNRsum = 0;
    double eventR = 0;
    double eventNR = 0;

    // Compute hit rates from counters
        for (int age = MAX_AGE_VALUE; age > 0; age--) {
            hNRsum += hitCtrs[age][0];
            eventNR += hitCtrs[age][0] + evictionCtrs[age][0];
            hRsum += hitCtrs[age][1];
            eventR += hitCtrs[age][1] + evictionCtrs[age][1];
            hR[age] = hRsum/eventR;
            hNR[age] = hNRsum/eventNR;
        }

    // Calculate overall hit rate
    double h = (hRsum + hNRsum)/(eventR + eventNR);
    // doible lineGain = 1. * (hRsum+hNRsum) / (eventR + eventNR) / MAX_AGE_VALUE;
    // double events = 0;
    // double lineGain = 0;
   
    // double event[MAX_AGE_VALUE] = {0.};
    // double totalEventsAbove[MAX_AGE_VALUE] = {0.};
    
    // for (int age = 1; age <= MAX_AGE_VALUE-1; age++) {
    //     event[age] = ewmaHits[age] + ewmaEvictions[age];
    //     totalEventsAbove[age] = totalEventsAbove[age+1] + event[age];
    //     lineGain += 1. * ewmaHits[age] / (ewmaHits[age] + ewmaEvictions[age]) / MAX_AGE_VALUE;
    // }
    // double lineGain = 1. * ewmaHits / (ewmaHits + ewmaEvictions) / MAX_AGE_VALUE;
    double perAccessCost = h * A / (NUM_SETS*NUM_WAYS);

    // Compute EVA
    std::vector<std::vector<double>> eva(2, std::vector<double>(2 * (1 << (k + 1)), 0));

    for (int c = 0; c < 2; c++) {
         double expectedLifetime = 0; // Initialize expectedLifetime to 0
         double hits = 0; // Initialize hits to 0
         double events = 0; // Initialize events to 0
        // double expectedLifetimeUnconditioned = 0;
        for (int age = MAX_AGE_VALUE; age > 0; age--) {
            expectedLifetime += events;
            // expectedLifetimeUnconditioned = (age/2+0.5) * totalEventsAbove[age];
            // expectedLifetime = ((1./6) * (age/2+0.5)* event[age] + expectedLifetimeUnconditioned) / (0.5 * event[age] + totalEventsAbove[age+1]);
            // eva[c][age] = (hNR[age] - lineGain * expectedLifetime) / eventNR[age];
            // expectedLifetime += eventR[age];
            // eva[c][age] += (hR[age] - lineGain * expectedLifetime) / eventR[age];
            eva[c][age] = (hits - perAccessCost * expectedLifetime) / events;
            hits += hitCtrs[age][c];
            events += hitCtrs[age][c] + evictionCtrs[age][c];
        }
    }
    double evaReused = eva[1][1] / (1 - hR[1]);
    // Differentiate classes
    for (int c = 0; c < 2; c++) {
        for (int age = MAX_AGE_VALUE; age > 0; age--) {
            if (c==0){
                eva[c][age] += (hNR[age] - h) * evaReused;
            }
            else{
                eva[c][age] += (hR[age] - h) * evaReused;   
            }
        }
    }

    // Finally, rank ages by EVA
    std::vector<std::vector<int>> order(2, std::vector<int>(2 * (1 << (k + 1))));
    for (int c = 0; c < 2; ++c) {
        std::iota(order[c].begin(), order[c].end(), 0);
        std::sort(order[c].begin(), order[c].end(), [&](int a, int b) {
            return eva[c][a] > eva[c][b];
        });
    }
    
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < MAX_AGE_VALUE; ++i) {
            rank[order[c][i]] = MAX_AGE_VALUE - i;
        }
    }

    return rank;
}

Cache_Age cache_age;

void CACHE::initialize_replacement() { 
}

uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type)
{
    // Compute EVA and update ranks
    double N = 8;
    std::vector<double> rank = computeEVAandUpdateRanks(hitCtrs_nn, evictionCtrs_nn, AGE_GRANULARITY, N);
    
    // Find the way with the maximum rank among all cache lines in the set
    double maxRank = std::numeric_limits<double>::min();
    uint32_t victimIndex = 0;
    for (int way = 0; way < NUM_WAYS; ++way) {
        auto [ageCounter, classificationBit, timestamp] = cache_age.getCacheLineInfo(set, way);
        double lineRank = rank[ageCounter];
        if (lineRank > maxRank) {
            maxRank = lineRank;
            victimIndex = way;
        }
    }

    // Update eviction counters for the selected victim
    auto [ageCounter, classificationBit, timestamp] = cache_age.getCacheLineInfo(set, victimIndex);
    evictionCtrs_nn[ageCounter][classificationBit]++;
    cache_age.accessCache(set, victimIndex, false, true); // Access set 0, way 1, with a hit, evict, call after all calc done
    
    return victimIndex;
}

void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
   // Get cache line info
  auto [ageCounter, classificationBit, timestamp] = cache_age.getCacheLineInfo(set, way);
  bool hit_age = false; 
  if(hit && access_type{type} != access_type::WRITE){ // If hit and skip for witeback hits
      auto [ageCounter, classificationBit, timestamp] = cache_age.getCacheLineInfo(set, way);
      hitCtrs_nn[ageCounter][classificationBit]++; // increment R
      hit_age = true;
  }
  cache_age.accessCache(set, way, hit_age, false); // Access set 0, way 1, with a hit, evict, call after all calc done

//   if (evafg == 10 && hit && access_type{type} != access_type::WRITE){
//     //   cache_age.printCacheStatus();
//     //   std::cout << "Value of entire hit0:evafg: " << evafg << std::endl; 
//       for (int f =0; f<32;f++){
//         std::cout << "Value of entire hit0: " << hitCtrs_nn[f][ageCounter][0] << std::endl; 
//         std::cout << "Value of entire hit: " << hitCtrs_nn[f][ageCounter][1] << std::endl; 
//       }
//   }
//   if (evafg <= 10 && hit && access_type{type} != access_type::WRITE){
//       evafg++;
//   }
    double N = 8;
    std::vector<double> rank = computeEVAandUpdateRanks(hitCtrs_nn, evictionCtrs_nn, AGE_GRANULARITY, N);
    // if (evafg == 10 && hit && access_type{type} != access_type::WRITE){
    // std::cout << "Ranking:\n";
    // for (double i = 0; i <= (2 * (1 << (k + 1)))-1; i++) {
    //     std::cout << "Age " << i << ": " << rank[i] << std::endl;
    // }
    //   }
  }

void CACHE::replacement_final_stats() {}
