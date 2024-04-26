#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include <algorithm>

namespace {
// const uint32_t k = 4; // Define the constant 2^k for age granularity
const int NUM_SETS = 32;        // Number of sets in the cache
const int NUM_WAYS = 16;        // Number of ways (cache lines) per set
const int AGE_BITS = 4;         // Number of bits for age counter
const int TIMESTAMP_BITS = 7;   // Number of bits for timestamp counter
const int AGE_GRANULARITY = 15; // Age granularity (limit A)
const int MAX_AGE_VALUE =
    (1 << (AGE_BITS + 1)); // Maximum value for age counter
const int PERIOD_UPDATES = 5;  // Number of bits for timestamp counter
// std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;
double hitCtrs_nn[MAX_AGE_VALUE][2] = {{0.0}};
double evictionCtrs_nn[MAX_AGE_VALUE][2] = {{0.0}};

double absl_hitCtrs_nn[MAX_AGE_VALUE][2] = {{0.0}};
double absl_evictionCtrs_nn[MAX_AGE_VALUE][2] = {{0.0}};
// Compute EVA
std::vector<std::vector<double>>
    G_eva(2, std::vector<double>(MAX_AGE_VALUE, 0));
// int evafg = 0;
double absl_decay = 0.8;
} // namespace

void print_G_eva() {

  std::cout << "\n////////////////////////////////////////////\n" << std::endl;
  std::cout << "age\t\t"
            << "evaNR:\t\t\t"
            << "evaR:\t" << std::endl;
  for (int f = 0; f < fmin(MAX_AGE_VALUE, 16); f++) {
    std::cout << f << "\t\t" << G_eva[0][f] << "\t\t" << G_eva[1][f]
              << std::endl;
  }
  std::cout << "\n////////////////////////////////////////////\n" << std::endl;
}

void print_hitCtrs_nn() {

  std::cout << "\n////////////////////////////////////////////\n" << std::endl;
  std::cout << "age\t\t"
            << "hit0:\t"
            << "hit1:\t"
            << "evict0:\t"
            << "evict1:\t" << std::endl;
  for (int f = 0; f < fmin(MAX_AGE_VALUE, 50); f++) {
    std::cout << f << "\t\t" << hitCtrs_nn[f][0] << "\t\t" << hitCtrs_nn[f][1]
              << "\t\t" << evictionCtrs_nn[f][0] << "\t\t"
              << evictionCtrs_nn[f][1] << std::endl;
  }
  std::cout << "\n////////////////////////////////////////////\n" << std::endl;
}

void print_absl_hit_evictCtrs_nn() {

  std::cout << "\n////////////////////////////////////////////\n" << std::endl;
  std::cout << "age\t\t"
            << "absl_hit0:\t"
            << "absl_hit1:\t"
            << "absl_evict0:\t"
            << "absl_evict1:\t" << std::endl;
  for (int f = 0; f < fmin(MAX_AGE_VALUE, 50); f++) {
    std::cout << f << "\t\t" << absl_hitCtrs_nn[f][0] << "\t\t" << absl_hitCtrs_nn[f][1]
              << "\t\t" << absl_evictionCtrs_nn[f][0] << "\t\t"
              << absl_evictionCtrs_nn[f][1] << std::endl;
  }
  std::cout << "\n////////////////////////////////////////////\n" << std::endl;
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
    CacheSet &currentSet = sets[set];
    currentSet.timestamp =
        (currentSet.timestamp.to_ulong() + 1) % AGE_GRANULARITY;

    for (auto &line : currentSet.lines) {
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
  std::tuple<unsigned int, bool, unsigned int> getCacheLineInfo(int set,
                                                                int way) {
    CacheSet &currentSet = sets[set];
    CacheLine &line = currentSet.lines[way];

    return std::make_tuple(line.ageCounter.to_ulong(), line.classificationBit,
                           currentSet.timestamp.to_ulong());
  }

  void printCacheStatus(int set, int way) {
    CacheSet &currentSet = sets[set];
    CacheLine &line = currentSet.lines[way];

    std::cout << "\tSet: " << set << ", Way: " << way
              << "\tAge Counter: " << line.ageCounter.to_ulong()
              << "\tClassification Bit: " << line.classificationBit
              << "\tTimestamp: " << currentSet.timestamp << std::endl;
  }

  void printCacheStatus(int set) {
    CacheSet &currentSet = sets[set];
    std::cout << "Set: " << set << std::endl;
    for (int way = 0; way < NUM_WAYS; way++) {
      CacheLine &line = currentSet.lines[way];
      std::cout << "Way: " << way
                << ", Age Counter: " << line.ageCounter.to_ulong()
                << ", Classification Bit: " << line.classificationBit
                << std::endl;
    }
    std::cout << "Timestamp: " << currentSet.timestamp.to_ulong() << std::endl;
    std::cout << std::endl;
  }

  void printCacheStatus() {

    std::cout << "\n////////////////////////////////////////////\n";
    for (int set = 0; set < NUM_SETS; set++) {
      CacheSet &currentSet = sets[set];

      std::cout << "Set: " << set << std::endl;
      for (int way = 0; way < NUM_WAYS; way++) {
        CacheLine &line = currentSet.lines[way];
        std::cout << "Way: " << way
                  << ", Age Counter: " << line.ageCounter.to_ulong()
                  << ", Classification Bit: " << line.classificationBit
                  << std::endl;
      }
      std::cout << "Timestamp: " << currentSet.timestamp.to_ulong()
                << std::endl;
      std::cout << std::endl;
    }
    std::cout << "\n////////////////////////////////////////////\n";
  }
};

void update_absl_hit_evit(){
for (int age = MAX_AGE_VALUE -1; age >= 0; age--) {
  absl_hitCtrs_nn[age][0] *= absl_decay;
  absl_hitCtrs_nn[age][0] += hitCtrs_nn[age][0];
  absl_hitCtrs_nn[age][1] *= absl_decay;
  absl_hitCtrs_nn[age][1] += hitCtrs_nn[age][1];
  // eviction 
  absl_evictionCtrs_nn[age][0] *= absl_decay;
  absl_evictionCtrs_nn[age][0] += evictionCtrs_nn[age][0];
  absl_evictionCtrs_nn[age][1] *= absl_decay;
  absl_evictionCtrs_nn[age][1] += evictionCtrs_nn[age][1];
    }

}
void reset_hit_evict(){
  for (int age = MAX_AGE_VALUE -1; age >= 0; age--) {
    evictionCtrs_nn[age][0] = 0;
    hitCtrs_nn[age][0] = 0;
    evictionCtrs_nn[age][1] = 0;
    hitCtrs_nn[age][1] = 0;
  }
 
}
// Function to compute EVA and update ranks
std::vector<std::vector<double>>
computeEVAandUpdateRanks(double hitCtrs[MAX_AGE_VALUE][2],
                         double evictionCtrs[MAX_AGE_VALUE][2], double A,
                         double N) {
  // Initialize variables
  std::vector<double> rank(MAX_AGE_VALUE);
  std::vector<double> hR(MAX_AGE_VALUE);
  std::vector<double> hNR(MAX_AGE_VALUE);
  double hRsum = 0;
  double hNRsum = 0;
  double eventR = 0;
  double eventNR = 0;
  double absl_hitCtrs_nn_sum =0;
  double absl_evictionCtrs_nn_sum =0;
  

  update_absl_hit_evit();
  
  // Compute hit rates from counters
  for (int age = MAX_AGE_VALUE -1; age >= 0; age--) {
    hNRsum += hitCtrs[age][0];
    eventNR += hitCtrs[age][0] + evictionCtrs[age][0];
    hRsum += hitCtrs[age][1];
    eventR += hitCtrs[age][1] + evictionCtrs[age][1];
    //absl sum values
    absl_hitCtrs_nn_sum += absl_hitCtrs_nn[age][0];
    absl_hitCtrs_nn_sum += absl_hitCtrs_nn[age][1];
    absl_evictionCtrs_nn_sum += absl_evictionCtrs_nn[age][0];
    absl_evictionCtrs_nn_sum += absl_evictionCtrs_nn[age][1];
    

    if (eventR == 0) {
      hR[age] = 0;    //to avoid nan
    } else {
      hR[age] = hRsum / eventR;
    }
    if (eventNR == 0) {
      hNR[age] = 0;
    } else {
      hNR[age] = hNRsum / eventNR;
    }
    // std::cout << "Value of eventRs" << eventR << "Hr " << hR[age] <<
    // std::endl; std::cout << "Value of eventNRs" << eventNR << "Hnr " <<
    // hNR[age] << std::endl;
  }

  // Calculate overall hit rate
  double h = (hRsum + hNRsum) / (eventR + eventNR);
  std::cout << "h: " << h
    << "\thRsum: " << hRsum
    << "\t, hNRsum: " << hNRsum
    << "\t, eventR: " << eventR
    << "\t, eventNR: " << eventNR
    << std::endl;
  std::cout << "\absl_hitCtrs_nn_sum: " << absl_hitCtrs_nn_sum
    << "\t, absl_evictCtrs_nn_sum: " << absl_evictionCtrs_nn_sum
    << std::endl;
  // doible lineGain = 1. * (hRsum+hNRsum) / (eventR + eventNR) / MAX_AGE_VALUE;
  // double events = 0;
  // double lineGain = 0;

  // double event[MAX_AGE_VALUE] = {0.};
  // double totalEventsAbove[MAX_AGE_VALUE] = {0.};

  // for (int age = 1; age <= MAX_AGE_VALUE-1; age++) {
  //     event[age] = ewmaHits[age] + ewmaEvictions[age];
  //     totalEventsAbove[age] = totalEventsAbove[age+1] + event[age];
  //     lineGain += 1. * ewmaHits[age] / (ewmaHits[age] + ewmaEvictions[age]) /
  //     MAX_AGE_VALUE;
  // }
  // double lineGain = 1. * hRsum / (hRsum + ewmaEvictions)
  // MAX_AGE_VALUE;
  A = absl_hitCtrs_nn_sum / (absl_hitCtrs_nn_sum + absl_evictionCtrs_nn_sum) / (NUM_SETS * NUM_WAYS);
  std::cout << "\t Age_granularoty: " << A
  << std::endl;
  double perAccessCost = h * A / (NUM_SETS * NUM_WAYS);

  // Compute EVA
  std::vector<std::vector<double>> eva(
      2, std::vector<double>(MAX_AGE_VALUE, 0));

  for (int c = 0; c < 2; c++) {
    double expectedLifetime = 0; // Initialize expectedLifetime to 0
    double hits = 0;             // Initialize hits to 0
    double events = 0;           // Initialize events to 0
    // double expectedLifetimeUnconditioned = 0;
    for (int age = MAX_AGE_VALUE; age > 0; age--) {
      expectedLifetime += events;
      // expectedLifetimeUnconditioned = (age/2+0.5) * totalEventsAbove[age];
      // expectedLifetime = ((1./6) * (age/2+0.5)* event[age] +
      // expectedLifetimeUnconditioned) / (0.5 * event[age] +
      // totalEventsAbove[age+1]); eva[c][age] = (hNR[age] - lineGain *
      // expectedLifetime) / eventNR[age]; expectedLifetime += eventR[age];
      // eva[c][age] += (hR[age] - lineGain * expectedLifetime) / eventR[age];
      if (events == 0) {
        eva[c][age] = 0;
      } else {
        eva[c][age] = (hits - perAccessCost * expectedLifetime) / events;
      }

      //  std::cout << "Value of c and a" << c << " , " << age << "EVA " <<
      //  eva[c][age] << std::endl;
      hits += hitCtrs[age][c];
      events += hitCtrs[age][c] + evictionCtrs[age][c];
    }
  }
  double evaReused = eva[1][1] / (1 - hR[1]);
  // Differentiate classes
  for (int c = 0; c < 2; c++) {
    for (int age = MAX_AGE_VALUE; age >= 0; age--) {
      if (c == 0) {
        eva[c][age] += (hNR[age] - h) * evaReused;
      } else {
        eva[c][age] += (hR[age] - h) * evaReused;
      }
    }
  }

  // Finally, rank ages by EVA
  std::vector<std::vector<int>> order(
      2, std::vector<int>(MAX_AGE_VALUE));
  for (int c = 0; c < 2; ++c) {
    std::iota(order[c].begin(), order[c].end(), 0);
    std::sort(order[c].begin(), order[c].end(),
              [&](int a, int b) { return eva[c][a] > eva[c][b]; });
  }

  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < MAX_AGE_VALUE; ++i) {
      rank[order[c][i]] = MAX_AGE_VALUE - i;
    }
  }
  // Assign g_eva with eva
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < MAX_AGE_VALUE; ++j) {
      G_eva[i][j] = eva[i][j];
    }
  }
  return eva;
}

Cache_Age cache_age;

uint32_t find_victim(uint32_t set) {
  // Find the way with the maximum rank among all cache lines in the set
  double maxRank = 0;
  uint32_t victimIndex = 0;
  // std::cout << "Its a HIT, printing set age counter";
  // cache_age.printCacheStatus( set );
  //  if (evafg == 1000){
  //     for (const auto& row : evaF) {
  //         for (const auto& elem : row) {
  //             std::cout << elem << " ";
  //         }
  //         std::cout << "\n";
  //     }

  // }
  unsigned int ageCounter;
  bool classificationBit;
  unsigned int timestamp;
  for (int way = 0; way < NUM_WAYS; ++way) {
    std::tie(ageCounter, classificationBit, timestamp) =
        cache_age.getCacheLineInfo(set, way);
    double lineRank = G_eva[classificationBit][ageCounter];
    if (lineRank > maxRank) {
      maxRank = lineRank;
      victimIndex = way;
    }
  }
  // std::cout << "Victim INDEX: " << victimIndex <<"  line rank: " << maxRank;

  // Update eviction counters for the selected victim
  std::tie(ageCounter, classificationBit, timestamp) =
      cache_age.getCacheLineInfo(set, victimIndex);
  evictionCtrs_nn[ageCounter][classificationBit]++;
  cache_age.accessCache(
      set, victimIndex, false,
      true); // Access set 0, way 1, with a hit, evict, call after all calc done

  return victimIndex;
}

void update_replacement_state(uint32_t set, uint32_t way, uint8_t hit) {
  unsigned int ageCounter;
  bool classificationBit;
  unsigned int timestamp;
  // Get cache line info
  std::tie(ageCounter, classificationBit, timestamp) =
      cache_age.getCacheLineInfo(set, way);

  bool hit_age = false;
  if (hit) { // If hit and skip for witeback hits
    std::tie(ageCounter, classificationBit, timestamp) =
        cache_age.getCacheLineInfo(set, way);
    hitCtrs_nn[ageCounter][classificationBit]++; // increment R
    hit_age = true;
  }
  cache_age.accessCache(set, way, hit_age,
                        false); // Access set 0, way 1, with a hit, evict, call
                                // after all calc done

  //   if (hit && access_type{type} != access_type::WRITE){
  //         std::cout << "Its a HIT, printing set age counter";
  //         cache_age.printCacheStatus( set );
  //     //   std::cout << "Value of entire hit0:evafg: " << evafg << std::endl;
  //   }

  // if (evafg == 10){
  // for (const auto& row : evaF) {
  //     for (const auto& elem : row) {
  //         std::cout << elem << " ";
  //     }
  //     std::cout << "\n";
  // }
  // }
}

// if (evafg == 10 && hit && access_type{type} != access_type::WRITE){
// std::cout << "Ranking:\n";
// for (double i = 0; i <= (MAX_AGE_VALUE)-1; i++) {
//     std::cout << "Age " << i << ": " << rank[i] << std::endl;
// }
//   }

void replacement_final_stats() {}

int main() {
  std::ifstream inputFile("cache_accesses.txt");
  std::string line;

  if (!inputFile.is_open()) {
    std::cerr << "Error: Unable to open input file." << std::endl;
    return 1;
  }
  int accessCount = 0;
  while (getline(inputFile, line) && accessCount < 200) {
    std::istringstream iss(line);
    int set, way;
    std::string action;

    iss >> set >> way >> action;
    // cache_age.printCacheStatus(set, way);
    if (action == "hit") {
      update_replacement_state(set, way, 1);
      std::cout << "Cache hit at set " << set << ", way " << way << std::endl;
    } else if (action == "evict") {
      uint32_t victimIndex = find_victim(set);
      std::cout << "Cache evict at set " << set << ", way " << victimIndex
                << std::endl;
    }
    if (accessCount % 10 == 0 && accessCount >= 100) {
      // print_hitCtrs_nn();
    }
    accessCount++;

    if (accessCount % PERIOD_UPDATES == 4) {
      // Compute EVA and update ranks
      double N = 8;
      std::vector<std::vector<double>> evaF = computeEVAandUpdateRanks(
          hitCtrs_nn, evictionCtrs_nn, AGE_GRANULARITY, N);
      print_absl_hit_evictCtrs_nn();
      print_hitCtrs_nn();
      // print_evictCtrs_nn();
      print_G_eva();
      // reset_hit_evict();
      // print_hitCtrs_nn();
    }
  }

  printf("MAX AGE: %d",MAX_AGE_VALUE);
  inputFile.close();

  return 0;
}
