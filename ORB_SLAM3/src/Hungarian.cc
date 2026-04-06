#include "Hungarian.h"
#include <vector>
#include <limits>
#include <algorithm>

namespace ORB_SLAM3
{
    std::vector<int> Hungarian::solve(const std::vector<std::vector<float>>& cost)
    {
        int n = cost.size();
        int m = cost[0].size();
        int size = std::max(n, m);

        std::vector<std::vector<float>> a(size, std::vector<float>(size, 1e6));

        // Copy cost into square matrix
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                if (cost[i][j] < 0) 
                    continue;
                a[i][j] = cost[i][j];
            }
        }

        // u[i] potential for row i
        // v[j] potential for column j
        std::vector<float> u(size+1), v(size+1);

        // p[j] which row is matched to column j
        // way[j] path reconstruction (like parent pointer)
        std::vector<int> p(size+1), way(size+1);

        for(int i = 1; i <= size; i++)
        {
            // i is matched to column 0
            p[0] = i;
            int j0 = 0;
            // minv[j] = best reduced cost to reach column j
            std::vector<float> minv(size+1, std::numeric_limits<float>::max());
            std::vector<char> used(size+1, false);

            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                float delta = std::numeric_limits<float>::max();

                for(int j = 1; j <= size; j++)
                {
                    if(!used[j])
                    {
                        // Adjusted cost = real cost - potentials
                        float cur = a[i0-1][j-1] - u[i0] - v[j];
                        if(cur < minv[j])
                        {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if(minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for(int j = 0; j <= size; j++)
                {
                    if(used[j])
                    {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                        minv[j] -= delta;
                }

                j0 = j1;
            } while(p[j0] != 0);

            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while(j0);
        }

        std::vector<int> assignment(n, -1);

        for(int j = 1; j <= size; j++)
        {
            if(p[j] <= n && j <= m)
                assignment[p[j]-1] = j-1;
        }

        return assignment;
    }
}