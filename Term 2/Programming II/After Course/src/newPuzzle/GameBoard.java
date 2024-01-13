package newPuzzle;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class GameBoard extends JFrame {
	
	
	public GameBoard() {
		
		this.setSize(1150,680);
		this.setResizable(false);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		newPuzzle.PuzzlePanel panel = new newPuzzle.PuzzlePanel();
		
		this.add(panel);
		
		this.setVisible(true);

		
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		new GameBoard();
	}
	
	

}
