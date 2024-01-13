package prog1project;
import java.util.Scanner;

public class tictactoe {
	
	public static void main(String[] args) {
		
		Scanner console = new Scanner(System.in);
		
		char[][] game_board = {{'1','2','3'},{'4','5','6'},{'7','8','9'}}; //2D char array to act as game board
		
		System.out.println("Let's play Tic Tac Toe!");
		System.out.println("What's your name?");
		System.out.print("Player 1 \"X\": ");
		String player1 = console.next();
		System.out.println(player1 + ", you'll be playing as 'X'.");
		System.out.print("Player 2 \"O\": ");
		String player2 = console.next();
		System.out.println(player2 + ", you'll be playing as 'O'.");
		
		
		boolean xturn = true;	//boolean to store if it's player1's or player2's turn
		int count = 0;			//a counter to count the number of turns
		boolean winner = false;	//a boolean to store if the game is over or not
		
		//the main game loop, that keeps going until there's a winner, or until 9 turns have passed
		do {
			count++;	//incrementing the count with each turn (loop)
			
			char player_symbol;
			if(xturn) {		//setting whether the player's symbol is x or y depending on which player's turn it is
				player_symbol = 'X';
			}
			else {
				player_symbol = 'O';
			}
			
			
			printboard(game_board);	//printing game board using method created below
			
			if(xturn) {		//asking user to choose a place
					System.out.print(player1 + "'s turn. Choose a place: ");
			}
			else {
				System.out.print(player2 + "'s turn. Choose a place: ");
			}

			int row = 0, col = 0;	/*to store row and column number of place user chose,
										based on the following switch statement*/
			do {
				int place = console.nextInt();	//using the scanner to allow the user to input a place

				switch(place) {
				
					case 1:
						row = 0;
						col = 0;
						break;
					case 2:
						row = 0;
						col = 1;
						break;
					case 3:
						row = 0;
						col = 2;
						break;
					case 4:
						row = 1;
						col = 0;
						break;
					case 5:
						row = 1;
						col = 1;
						break;
					case 6:
						row = 1;
						col = 2;
						break;
					case 7:
						row = 2;
						col = 0;
						break;
					case 8:
						row = 2;
						col = 1;
						break;
					case 9:
						row = 2;
						col = 2;
						break;
				}
				//checking if the number is invalid or the place is taken
				if(place < 1 || place > 9) {
					System.out.println("Invalid input. Please enter a number from 1 to 9.");
				}
				else if(game_board[row][col] == 'X' || game_board[row][col] == 'O') {
					System.out.println("Place taken. Please choose another place.");
				}
				else {
					break;
				}
			} while(true);
				
			
			game_board[row][col] = player_symbol;	//setting the place the player chose to have their symbol
			
			
			//checking for complete rows, columns, or diagonals
			if(count>=5) {
				int test_row, test_col, test_diagmain, test_diagsec;
				test_row = test_col = test_diagmain = test_diagsec = 0;
				/*using nested loop to check if we have 3 of the user's symbol in the same
				 row, column, or diagonal, which means the user has won*/
				for(int i = 0; i < 3; i++) {	
					test_row = test_col = 0;
					for(int j = 0; j <3; j++) {
						if(game_board[i][j] == player_symbol ) {
							test_row++;
						}
						if (game_board[j][i] == player_symbol) {
							test_col++;
						}
					}
					if(test_row == 3 || test_col == 3) {
						winner = true;
						break;	
					}
					
					if(game_board[i][i] == player_symbol) {
						test_diagmain++;
					}
					if(game_board[i][2-i] == player_symbol) {
						test_diagsec++;
					}
				}
				if(test_diagmain == 3 || test_diagsec == 3) {
					winner = true;	
				}
			}
				
			//setting the winner as either player 1 or player 2
			if(winner && xturn) {
				printboard(game_board);
				System.out.println(player1 + " wins!");
				break;
			}
			else if(winner && !xturn) {
				printboard(game_board);
				System.out.println(player2 + " wins!");
				break;
			}
			else if(count == 9) { 		//ending the game after 9 turns
				printboard(game_board);
				System.out.println("Tie!");
				break;
			}
			xturn = !xturn;
		} while(true);
		
	}

	
	//method to print board
	public static void printboard(char[][] board) {
		System.out.println();
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				System.out.print(board[i][j]);
				if(j == 2) {
					continue;
				}
				System.out.print(" | ");
			}
			System.out.println();
			System.out.println();
		}
	}
	
}


