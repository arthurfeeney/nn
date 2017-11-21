
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <ctime> // gross but more convenient than chrono :(
#include "include/ensemble.hpp"
#include <utility>

using std::cout;
using std::stoi;
using std::vector;
using std::rand;
using std::generate;


/*
 *This is going to be really gross since I just want this to be a simple test 
 *for the network. Not important enough to warrant really making things clear.
 */


struct RandIntInRange {
    std::random_device rd;
    std::default_random_engine e;
    std::uniform_int_distribution<size_t> dist;
    size_t min;
    size_t max;

    RandIntInRange(size_t min, const size_t& max):
        rd(),
        e(rd()),
        dist(min, max),
        min(min),
        max(max)
    {}

    RandIntInRange(const RandIntInRange& other):
        rd(),
        e(other.e),
        dist(other.min, other.max),
        min(other.min),
        max(other.max)
    {}

    int operator()() {
        return dist(e); 
    }
};

vector<vector<int>> init_board(size_t dim, size_t max_val) {
    vector<vector<int>> board(dim, vector<int>(dim, 0));

    RandIntInRange r(0, max_val);
    for(auto& row : board) {
        for(auto& val : row) {
            val = r();
        }
    }

    return board;
}

class GameSpace {
    // doesn't need to be a class, but will be easier to manage stuff.
private:    

    class Enemy {
    private:
        std::pair<size_t, size_t> pos;
        char move_to_make;
    public:
        Enemy() {}

        Enemy(size_t board_size, RandIntInRange r) {
            pos = std::make_pair(static_cast<size_t>(r()), 
                                 static_cast<size_t>(r()));
        }

        Enemy& operator=(Enemy&& other) {
            if(this == &other) {
                return *this;
            }
            pos = other.pos;
            move_to_make = other.move_to_make;
            return *this;
        }

        void search_for_player() {
            // A*
            // put best move in move_to_make.
        }
        void make_move() {
            // make the best move found by A*
            // update pos based on move_to_make.
        }

        std::pair<size_t, size_t> get_pos() {
            return pos;
        }
    };
    vector<Enemy> enemies;
    vector<vector<int>> board;
    
    std::pair<size_t, size_t> player_pos;

    unsigned int player_score = 0;

public:
    GameSpace(vector<vector<int>> board, size_t number_enemies):
        board(board),
        enemies(number_enemies)
    {
        RandIntInRange r(0, this->board.size());
        // construct enemies.
        for(size_t i = 0; i < number_enemies; ++i) 
            enemies[i] = Enemy(board.size(), r);
        // set random player position.
        player_pos = std::make_pair(static_cast<size_t>(r()), 
                                    static_cast<size_t>(r()));
    }

    void step(char dir) {
        if(dir == 'w') {
            player_pos.second -= 1;
        }
        else if(dir == 'a') {
            player_pos.first -= 1;
        }
        else if(dir == 's') {
            player_pos.second += 1;
        }
        else if(dir == 'd') {
            player_pos.first += 1;
        }
        else {
            cout << "THAT IS AN INVALID DIRECTION!!!" << '\n';
        }

        player_score += board[player_pos.first][player_pos.second];

        for(auto& enemy : enemies) {
            enemy.search_for_player();
            enemy.make_move();
        }
    }
    
    bool good_board() {
        // returns true if play should continue. Otherwise, false!
        // if enemy is player, play should stop.
        for(auto& enemy : enemies) {
            auto exy = enemy.get_pos();
            if(exy.first == player_pos.first && exy.second == player_pos.second) {
                return false;
            }
        }
        return true;
    }

    void print_board() {
        std::cout << player_pos.first << ' ' << player_pos.second << '\n';
        for(size_t row = 0; row < board.size(); ++row) {
            for(size_t col = 0; col < board[0].size(); ++col) {
                std::string print = std::to_string(board[row][col]);
                if(player_pos.first == row && player_pos.second == col) {
                    print = "P";
                }
                for(auto& enemy : enemies) {
                    auto exy = enemy.get_pos();
                    if(exy.first == row && exy.second == col) {
                        print = "E";
                    }
                }
                std::cout << print << ' ';
            }
            cout << '\n';
        }
    }
};




int main(int argc, char** argv) {
    
    size_t board_size = 0;

    if(argc < 2) {
        board_size = 8;
    }
    else if(argc == 2) {
        board_size = stoi(argv[1]);
    }
    else {
        cout << "too many cl arguments. Only pass board_size." << '\n';
    }

    auto board = init_board(board_size, 9);
    
    GameSpace space(board, 2);
    space.print_board();

    return 0;
}
