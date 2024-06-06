clc;
clear;
close all;

% Definisi parameter algoritma
ps = 30;           % Population size (jumlah partikel)
phi = 2;           % Social factor
F = rand(1);       % Mutation factor
CR = 0.11;         % Crossover factor
max_iter = 100;    % Jumlah iterasi maksimum
dim = 10;          % Total jumlah pembangkit di 3 area (4+3+3)

% Batas bawah dan atas daya untuk setiap pembangkit
P_min = [196, 114, 200, 99, 190, 85, 200, 99, 130, 100]; % Batas minimum
P_max = [250, 157, 500, 265, 490, 265, 500, 265, 440, 500]; % Batas maksimum

% Batas daya transmisi antar area
T_min = [0, 0, 0]; % Batas minimum
T_max = [100, 100, 100]; % Batas maksimum

% Koefisien biaya bahan bakar
a = [0.01, 0.02, 0.015, 0.017, 0.018, 0.017, 0.016, 0.019, 0.02, 0.018];
b = [2.0, 1.8, 2.5, 2.3, 1.9, 2.1, 2.2, 2.0, 1.85, 2.05];
c = [100, 120, 130, 110, 115, 125, 135, 105, 112, 122];
e = [0.1, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2, 0.15, 0.1, 0.2];
f = [0.05, 0.04, 0.06, 0.05, 0.04, 0.06, 0.05, 0.04, 0.06, 0.05];

% Parameter MASED
total_demand = 2700; % Total permintaan daya listrik (MW)
demand = [0.25, 0.5, 0.25] * total_demand; % Distribusi permintaan per area (MW)

% Indeks pembangkit per area
area_indices = {1:4, 5:7, 8:10};

% Fungsi biaya bahan bakar per pembangkit dengan penalti
cost_func = @(P) calculate_cost_with_penalty(P, a, b, c, e, f, demand, area_indices, P_min, P_max, T_min, T_max);

% Inisialisasi posisi dan kecepatan awal partikel
pos = P_min + (P_max - P_min) .* rand(ps, dim);
vel = zeros(ps, dim);

% Inisialisasi nilai fitness
fitness = zeros(ps, 1);
for i = 1:ps
    fitness(i) = cost_func(pos(i, :));
end

% Inisialisasi posisi dan fitness terbaik individu dan global
pbest = pos;
pbest_fitness = fitness;
[gbest_fitness, gbest_idx] = min(pbest_fitness);
gbest = pbest(gbest_idx, :);

% Set waktu ke 1, dan fitness evaluation ke 0
t = 1;
FES = 0;

% Algoritma ImCSO
while FES < max_iter * ps
    % pecah populasi menjadi GW dan GL
    [~, sorted_idx] = sort(fitness);
    GW = sorted_idx(1:floor(ps/2));
    GL = sorted_idx(floor(ps/2) + 1:end);

    % Update GL particles (hanya update kecepatan, posisi, fitness, dan global fitness)
    for i = GL'
        % Update velocity
        R1 = rand;
        R2 = rand;
        R3 = rand;
        gbest_index = GW(randi(length(GW))); % Randomly pick from GW
        X_center = mean(pos(GW, :)); % Calculate center of GW
        vel(i, :) = R1 * vel(i, :) + R2 * (pos(gbest_index, :) - pos(i, :)) + R3 * phi * (X_center - pos(i, :));
        
        % Update position
        pos(i, :) = pos(i, :) + vel(i, :);
        pos(i, :) = max(min(pos(i, :), P_max), P_min);

        % Evaluasi solusi
        trial_fitness = cost_func(pos(i, :));
        FES = FES + 1;
        
        % Update fitness
        if trial_fitness < fitness(i)
            fitness(i) = trial_fitness;
            if trial_fitness < pbest_fitness(i)
                pbest(i, :) = pos(i, :);
                pbest_fitness(i) = trial_fitness;
            end
        end
        
        % Update gbest
        if trial_fitness < gbest_fitness
            gbest_fitness = trial_fitness;
            gbest = pos(i, :);
        end
    end

    % Update GW particles (melakukan mutasi dan crossover)
    for i = GW'
        % Mutation
        idxs = randperm(ps, 3);
        while any(idxs == i)
            idxs = randperm(ps, 3);
        end
        mutant = pos(i, :) + F * (pbest(idxs(2), :) - pbest(idxs(3), :));
        mutant = max(min(mutant, P_max), P_min);
        
        % Crossover
        trial = pos(i, :);
        jrand = randi(dim); % Random dimension for crossover
        for j = 1:dim
            if rand < CR || j == jrand
                trial(j) = mutant(j);
            end
        end
        
        % Evaluasi solusi
        trial_fitness = cost_func(trial);
        FES = FES + 1;
        
        % Selection
        if trial_fitness < fitness(i)
            pos(i, :) = trial;
            fitness(i) = trial_fitness;
            if trial_fitness < pbest_fitness(i)
                pbest(i, :) = trial;
                pbest_fitness(i) = trial_fitness;
            end
        end
        
        % Update gbest
        if trial_fitness < gbest_fitness
            gbest_fitness = trial_fitness;
            gbest = trial;
        end
    end

    fprintf('%d\t\t%f\n', t, gbest_fitness);

    % Menyimpan nilai fitness terbaik
    best_fitness(t) = gbest_fitness;
    t = t + 1;
end
disp('=======================================');
disp(' ');

% Menampilkan hasil
figure;
plot(best_fitness, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness');
title('ImCSO Optimization');
grid on;

% Nilai solusi terbaik
disp('Best Solution:');
disp(gbest);
disp('Best Fitness Value:');
disp(gbest_fitness);

function cost = calculate_cost_with_penalty(P, a, b, c, e, f, demand, area_indices, P_min, P_max, T_min, T_max)
    % Constraint daya transmisi antar area
    T = zeros(1, 3);
    for area = 1:3
        area_demand = sum(P(area_indices{area}));
        T(area) = area_demand - demand(area);
        
        % Apply transmission limits
        if T(area) < T_min(area)
            T(area) = T_min(area);
        elseif T(area) > T_max(area)
            T(area) = T_max(area);
        end
    end
    
    % Terapkan constraint tambahan
    for i = 1:length(P)
        if P(i) <= P_min(i)
            P(i) = P_min(i);
        elseif P(i) >= P_max(i)
            P(i) = P_max(i);
        end
    end

    % Menghitung biaya bahan bakar
    cost = sum(a .* P.^2 + b .* P + c + e .* sin(f .* (P_min - P)));
    
    % Menghitung daya per area
    area_demand = zeros(1, length(area_indices));
    for area = 1:length(area_indices)
        area_demand(area) = sum(P(area_indices{area}));
    end

    % Constraint keseimbangan beban daya
    power_balance_penalty = 0;
    for area = 1:length(demand)
        diff = area_demand(area) - (demand(area) + sum(T) - T(area));
        if diff > 0
            % Jika daya berlebih, kurangi dari pembangkit secara acak
            j = randi(length(area_indices{area}));
            P_idx = area_indices{area}(j);
            P(P_idx) = max(P(P_idx) - diff, P_min(P_idx));
        elseif diff < 0
            % Jika daya kurang, tambah ke pembangkit secara acak
            j = randi(length(area_indices{area}));
            P_idx = area_indices{area}(j);
            P(P_idx) = min(P(P_idx) - diff, P_max(P_idx));
        end
        
        % Hitung penalti
        power_balance_penalty = power_balance_penalty + abs(diff);
    end
    
    % Penalti untuk batas daya transmisi antar area
    transmission_penalty = sum(max(0, T - T_max)) + sum(max(0, T_min - T));
    
    % Penalti keseluruhan
    penalty = 1000 * power_balance_penalty + 1000 * transmission_penalty;
    cost = cost + penalty;
end
