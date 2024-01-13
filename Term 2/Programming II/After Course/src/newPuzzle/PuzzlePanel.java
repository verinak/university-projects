package newPuzzle;

import java.awt.Rectangle;
import java.awt.Point;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;


import puzzle.PuzzlePiece;

public class PuzzlePanel extends JPanel {	//m7taga agarab a3mel el kalam da fl jframe 3latoul.. w momken a7ot el sowar f jlabels? bs m3raf4 hat3ok fl painting wla la2a
	
	private newPuzzle.PuzzlePiece imageClicked;
	Point imageCorner;
	Point prevPt;
	private newPuzzle.PuzzlePiece[][] gojo;
	private Rectangle[][] rectangles;
	private Rectangle imageRectangle;
	JLabel gridLabel;
	ImageIcon preview;
	JLabel previewLabel;
	JCheckBox showPreview; 
	JButton reload;
	private boolean dragEnabled;
	
	PuzzlePanel() {
		dragEnabled = true;
		this.setLayout(null);
		ImageIcon gridImage = new ImageIcon("gojou puzzle pieces/grid.png");
		gridLabel = new JLabel(gridImage);
		gridLabel.setBounds(0,0,600,600);
		this.add(gridLabel);
		
		preview = new ImageIcon("gojou puzzle pieces/lowopacity.png");
		previewLabel = new JLabel(preview);
		previewLabel.setBounds(0,0,600,600);
		showPreview = new JCheckBox("Show preview");
		showPreview.setBounds(170,610,105,20);
		this.add(showPreview);
		CheckBoxAction ca = new CheckBoxAction();
		showPreview.addActionListener(ca);
		
		reload = new JButton("Reload puzzle");
		reload.setBounds(305,608,150,25);
		this.add(reload);
		ButtonAction ba = new ButtonAction();
		reload.addActionListener(ba);
		
		gojo = new newPuzzle.PuzzlePiece[3][3];
		rectangles = new Rectangle[3][3];
		
		createPuzzle();
		
		
		MouseMotion ml = new MouseMotion();
		this.addMouseListener(ml);
		this.addMouseMotionListener(ml);
		
		
	}
	
	private void createPuzzle() {
		gojo[0][0] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a00.png", new Rectangle(-2,-2,4,4), new Point(0,0));
		gojo[0][1] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a01.png", new Rectangle(198,-2,4,4), new Point(200,0));
		gojo[0][2] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a02.png", new Rectangle(398,-2,4,4), new Point(400,0));
		gojo[1][0] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a10.png", new Rectangle(-2,198,4,4), new Point(0,200));
		gojo[1][1] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a11.png", new Rectangle(198,198,4,4), new Point(200,200));
		gojo[1][2] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a12.png", new Rectangle(398,198,4,4), new Point(400,200));
		gojo[2][0] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a20.png", new Rectangle(-2,398,4,4), new Point(0,400));
		gojo[2][1] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a21.png", new Rectangle(198,398,4,4), new Point(200,400));
		gojo[2][2] = new newPuzzle.PuzzlePiece("gojou puzzle pieces/a22.png", new Rectangle(398,398,4,4), new Point(400,400));
		
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				rectangles[i][j] = new Rectangle(200,200);
			}
		}
		
		/* int x = 0;
		int y = 0;
		for(int i = 0; i < 3; i++) {
			x = 0;
			for(int j = 0; j < 3; j++) {
				String address = "gojou puzzle pieces/a" + i + j+ ".png";
				gojo[i][j] = new newPuzzle.PuzzlePiece(address, new Point(x,y));
				x = x + 200;
			}
			y = y+200;
		}*/
	}
	
	//hena 3amalt override ll paint 34an ana 3yza a-paint fo2 el children(eli homa el grid wl preview)
	//paintcomponent btetnafez fl awel ba3den paintborder ba3den paintchildren f kano byeteb3o ta7taha
	@Override
	public void paint(Graphics g) {
		super.paint(g);
		//lazem a repaint el 9 pieces kol mara
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				newPuzzle.PuzzlePiece tempPiece = gojo[i][j];
				Point tempPoint = tempPiece.getPaintLocation();
				tempPiece.paintIcon(this, g, (int)tempPoint.getX(), (int)tempPoint.getY());
				rectangles[i][j].setLocation(tempPoint);
			}
		}
		if(imageClicked!=null) {
			imageRectangle.setLocation((int)imageCorner.getX(), (int)imageCorner.getY());
			imageClicked.paintIcon(this, g, (int)imageCorner.getX(), (int)imageCorner.getY());
		}
	}
	
	
	public class MouseMotion implements MouseListener, MouseMotionListener {

		@Override
		public void mouseDragged(MouseEvent e) {
			// TODO Auto-generated method stub
			if(dragEnabled & imageClicked!=null) {
					imageCorner = imageClicked.getPaintLocation();
					if(imageClicked.isMoveable()) {
						Point currentPt = e.getPoint();
						imageCorner.translate(
								(int)(currentPt.getX() - prevPt.getX()),
								(int)(currentPt.getY() - prevPt.getY())
								);
						prevPt = currentPt;
						repaint();
						
						if((imageClicked.getCornerRectangle().contains(imageCorner))) {
							imageCorner = imageClicked.getFinalLocation();
							imageClicked.setPaintLocation(imageCorner);
							repaint();
							imageClicked.setMoveable(false);
							imageRectangle.setSize(1,1);
							newPuzzle.PuzzlePiece.piecesPlaced++;
							if(newPuzzle.PuzzlePiece.piecesPlaced == 9) {
								JOptionPane.showMessageDialog(getParent(),"Puzzle complete!", "", JOptionPane.PLAIN_MESSAGE);
							}
						}
					}
			}
				
		}

		@Override
		public void mouseMoved(MouseEvent e) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void mouseClicked(MouseEvent e) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void mousePressed(MouseEvent e) {
			// TODO Auto-generated method stub
			prevPt = e.getPoint();
			for(int i = 2; i >= 0; i--) {
				for(int j = 2; j >= 0; j--) {
					if(rectangles[i][j].contains(e.getPoint())) {
						imageClicked = gojo[i][j];
						imageRectangle = rectangles[i][j];
						return;
					}
				}
			}
		}

		@Override
		public void mouseReleased(MouseEvent e) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void mouseEntered(MouseEvent e) {
			// TODO Auto-generated method stub
			dragEnabled = true;
		}

		@Override
		public void mouseExited(MouseEvent e) {
			// TODO Auto-generated method stub
			dragEnabled = false;
		}
		
	}
	
	public class CheckBoxAction implements ActionListener {

		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			if(showPreview.isSelected()) {
				add(previewLabel);
				repaint();
			}
			else {
				remove(previewLabel);
				repaint();
			}
		}
		
	}
	
	public class ButtonAction implements ActionListener {

		@Override
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			imageClicked = null;
			createPuzzle();
			newPuzzle.PuzzlePiece.piecesPlaced = 0;
			if(gridLabel.getParent() == null) {
				add(gridLabel);
			}
			repaint();
		}
		
	}

}
