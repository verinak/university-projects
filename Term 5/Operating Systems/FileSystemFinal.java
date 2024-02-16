import java.text.SimpleDateFormat;
import java.util.*;
import java.lang.Math;


class FileSystemObject {
    String name;
    int size;
    String location;
    String protection;
    String identifier;

    public FileSystemObject(String name, int size, String location, String protection) {
        this.name = name;
        this.size = size;
        this.location = location;
        this.protection = protection;
        this.identifier = this.toString();

    }
	

    public Folder getParentFolder(Folder root) {
        // get parent directory
        String path = this.location;
        Folder currFolder = root;
        for(String folderName : path.split("/")){
            FileSystemObject child = currFolder.getChild(folderName);
            if(child instanceof Folder) {
                currFolder = (Folder)child;
            }
        }
        return currFolder;
    }

    public Folder getParentPartition(Folder root) {
        // get parent directory
        String path = this.location;
        Folder currFolder = root;
        for(String folderName : path.split("/")){
            FileSystemObject child = currFolder.getChild(folderName);
            if(child instanceof Partition) {
                currFolder = (Folder)child;
            }
        }
        return currFolder;
    }

    public void updateProtection(String newProtection) {
        this.protection = newProtection;
    }
}

class File extends FileSystemObject {
    String extension;
    String content;
    Date dateTime;
    int blocks; // Number of blocks
    Date creationTime;
    Date modificationTime;
    Date accessTime;

    public File(String name, int size, String location, String protection,
                String extension, String content, Date dateTime, int blocks) {
        super(name, size, location, protection);
        this.extension = extension;
        this.content = content;
        this.dateTime = dateTime;
        this.blocks = blocks;
        this.creationTime = new Date(); // Set creation time to current time
        this.modificationTime = new Date(); // Set modification time to current time
        this.accessTime = new Date(); // Set access time to current time
    }
    // Update the access time whenever the file is read
    public void updateAccessTime() {
        this.accessTime = new Date();
    }

    // Update the modification time whenever the file is modified
    public void updateModificationTime() {
        this.modificationTime = new Date();
    }

    // @Override
    // public String toString() {
    //     SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    //     return "File: " + name +
    //             "\nLocation: " + location +
    //             "\nSize: " + size + " bytes" +
    //             "\nProtection: " + protection +
    //             "\nExtension: " + extension +
    //             "\nContent: " + content +
    //             "\nCreation Time: " + dateFormat.format(creationTime) +
    //             "\nModification Time: " + dateFormat.format(modificationTime) +
    //             "\nAccess Time: " + dateFormat.format(accessTime) +
    //             "\nBlocks: " + blocks;
    // }
    // Getter for modification time
    public Date getModificationTime() {
        return modificationTime;
    }

    // Getter for access time
    public Date getAccessTime() {
        return accessTime;
    }
    public void renameFile(String newName){
        // change name
        this.name = newName;
        // change location
        String[] pathList = (this.location).split("/");
        pathList[pathList.length - 1] = newName;
        this.location = String.join("/", pathList);
    }

    public File createCopy(Folder rootPartition) {
        File newFile = new File(this.name, this.size, this.location, this.protection, this.extension, this.content, new Date(),this.blocks);
        newFile.renameFile(this.name + "_copy");
        Folder parent = this.getParentFolder(rootPartition);
        int status = parent.addChild(newFile, rootPartition);
        if(status == -1) {
            return null;
        }
        return newFile;
    }
}

class Folder extends FileSystemObject {
    Map<String, FileSystemObject> children;

    public Folder(String name, int size, String location, String protection) {
        super(name, size, location, protection);
        this.size = size;
        this.children = new HashMap<>();
    }

    // public void addChild(FileSystemObject child, Folder rootPartition) {
    //     children.put(child.name, child);
    //     size += child.size;
    // }

    ////////////
    public void getDirectoryInformation() { 
        System.out.println("Directory Information for " + this.name + ":");
        System.out.println("Location: " + this.location);
        System.out.println("Size: " + this.size);
        System.out.println("Protection: " + this.protection);
        System.out.println("Number of Children: " + this.children.size());
    }

    public int addChild(FileSystemObject child, Folder rootPartition) {

        if(child.size != 0) {

            int freeSpace = 0;
            // lw ana ba-add direct fl partition
            if(this instanceof Partition) {
                freeSpace = ((Partition)this).maxSize - this.getUsedSize();
            }
            // 8er keda lazem arou7 agib awel partition 34an mayenfa34 a3ady el max size bta3ha
            else {
                Folder parent = this;
                while(true) {
                    parent = parent.getParentPartition(rootPartition);
                    if(parent instanceof Partition){
                        break;
                    }
                }
                freeSpace = ((Partition)parent).maxSize - parent.getUsedSize();
            }
            // law mafi4 makan
            if(freeSpace < child.size){
                System.out.println("Not enough space in target folder.");
                return -1;
            }
        }
        
        this.children.put(child.name, child);
        this.size += child.size;
        return 0;
        // matensi4 en momken nkoun bn8ot folder m4 file? bzat lw partition liha fixed size
        // momken akoun ba7ot 7aga b size 0 asln
    }
    
    // Override removeChild method to update size when removing a file or folder
    
    public void removeChild(String name) {
        FileSystemObject child = children.get(name);
        if (child != null) {
            this.children.remove(name);
            this.size -= child.size;
        }
    }

    public FileSystemObject getChild(String name) {
        return children.get(name);
    }

    public List<FileSystemObject> searchDirectory(String searchName) {
        List<FileSystemObject> searchResults = new ArrayList<>();
        searchDirectoryHelper(this, searchName, searchResults);
        return searchResults;
    }

    private void searchDirectoryHelper(Folder currentFolder, String searchName, List<FileSystemObject> searchResults) {
        for (FileSystemObject child : currentFolder.children.values()) {
            if ((child.name.toLowerCase()).contains(searchName.toLowerCase())) {
                searchResults.add(child);
                System.out.println(child.location);
            }
            if (child instanceof Folder) {
                // Recursive call for subfolders
                searchDirectoryHelper((Folder) child, searchName, searchResults);
            }
        }
    }

    public int getUsedSize() {
        int usedSize = 0;
        Folder currentFolder = this;

        for (FileSystemObject child : currentFolder.children.values()) {
            if (child instanceof File) {
                usedSize += child.size;
            }
            if (child instanceof Folder) {
                usedSize += ((Folder)child).getUsedSize();
            }
        }
        return usedSize;
    }

    public void listFiles() {
        System.out.println("Childs in " + this.name + ":");
        for (FileSystemObject child : children.values()) {
            if (child instanceof File) {
                System.out.println("- " + child.name);
            }
            else if (child instanceof Folder) {
            System.out.println("- Folder: " + child.name);
        }
        }
        if (children.isEmpty()) {
            System.out.println("Folder is empty.");
        }
    }

    public File navigateToFile(Scanner scanner) {

        Folder currFolder = this;

        while(true) {
            System.out.println();
            currFolder.listFiles();

            System.out.println("Enter folder name to navigate, or file name to select: ");
            String name = scanner.nextLine();
            
            FileSystemObject child = currFolder.getChild(name);
            
            if(child instanceof Folder){
                currFolder = (Folder)child;
            }
            else if (child instanceof File){
                return (File)child;
            }
            else {
                System.out.println("Invalid file/folder name.");
            }

        }
    }

    public Folder navigateToFolder(Scanner scanner) {
        Folder currFolder = this;

        while(true) {
            System.out.println();
            currFolder.listFiles();

            System.out.println("Enter folder name to navigate, or 'x' to select current folder: ");
            String name = scanner.nextLine();

            if(name.equalsIgnoreCase("x")){
                return currFolder;
            }
            else {
                FileSystemObject child = currFolder.getChild(name);
                if(child instanceof Folder){
                    currFolder = (Folder)child;
                }
                else {
                    System.out.println("Invalid folder name.");
                }
            }
            
        }

    }

    public void createFile(User currentUser, Scanner scanner, Folder rootPartition, int BLOCK_SIZE) {
        
        System.out.print("Enter the file name: ");
        String fileName = scanner.nextLine();
        System.out.print("Enter the file size: ");
        int fileSize = scanner.nextInt();
        scanner.nextLine(); // Consume the newline character
        System.out.print("Enter the file extension: ");
        String fileExtension = scanner.nextLine();
        System.out.print("Enter the file content: ");
        String fileContent = scanner.nextLine();

        // Create a new File object with the correct location
        String fileLocation = this.location + "/" + fileName;
        File newFile = new File(fileName, fileSize, fileLocation, "RW", fileExtension, fileContent, new Date(),(int)Math.ceil((float)fileSize/BLOCK_SIZE));
        
        for (FileSystemObject child : this.children.values()) {
            if (child instanceof File & child.name.equals(fileName)) {
                System.out.println("File name already exists.");
                return;
            }
        }


        // Add the new file to the selected folder
        int status = this.addChild(newFile, rootPartition);
        if(status != -1) {
            System.out.println("File created: " + newFile.name + " in folder: " + this.name);
        }
    }

    public void createFolder(User currentUser, Scanner scanner, Folder rootPartition) {
        // Prompt the user for folder details
        System.out.print("Enter the folder name: ");
        String folderName = scanner.nextLine();

        // Create a new Folder object
        Folder newFolder = new Folder(folderName, 0, this.location + "/" + folderName, "RWX");

        for (FileSystemObject child : this.children.values()) {
            if (child instanceof Folder & child.name.equals(folderName)) {
                System.out.println("Folder name already exists.");
                return;
            }
        }

        // Add the new folder to the current folder
        int status = this.addChild(newFolder, rootPartition);
        if(status != -1) {
        System.out.println("Folder created: " + newFolder.name);
        }
    }
        

}

class Partition extends Folder {
    int maxSize;
    int usedSize;

    public Partition(String name, int size, String location, String protection, int maxSize) {
        super(name, size, location, protection);
        this.maxSize = maxSize;
        this.usedSize = 0;
    }
    
    // Check if there is enough free space in the partition
    public boolean hasFreeSpace(int requiredSize) {
        return (maxSize - usedSize) >= requiredSize;
    }

    // Update used size
    public void updateUsedSize(int delta) {
        usedSize += delta;
    }
    
    
    @Override
    public void listFiles() {
        System.out.println("Files in " + this.name + ":");
        for (FileSystemObject child : children.values()) {
            if (child instanceof File) {
                System.out.println("- " + child.name);
            }
            else if (child instanceof Folder) {
                System.out.println("- Folder: " + child.name);
            }
        }
        if (children.isEmpty()) {
            System.out.println("Folder is empty.");
        }
    }


}

class User {
    String name;
    String[] allowedOperations;

    public User(String name, String[] allowedOperations) {
        this.name = name;
        this.allowedOperations = allowedOperations;
    }
    public String getName() {
        return name;
    }

    public boolean hasPermission(String operation) {
        for (String allowedOperation : allowedOperations) {
            if (allowedOperation.equals(operation)) {
                return true;
            }
        }
        return false;
    }
}




/////// MAIN //////////////////
public class FileSystemFinal {

    final static int BLOCK_SIZE = 20;

    public static void main(String[] args) {

        // Create a partition
        Partition rootPartition = new Partition("Root", 0, "/root", "RWX", 1000);

        Partition cDrive = new Partition("C:", 0, "/root/C:", "RWX", 300);
        Partition dDrive = new Partition("D:", 0, "/root/D:", "RWX", 700);
        rootPartition.addChild(cDrive, rootPartition);
        rootPartition.addChild(dDrive, rootPartition);

        // Create a folder
        Folder documents = new Folder("Documents", 0, "/root/C:/Documents", "RWX");

        // Add folder to partition
        cDrive.addChild(documents, rootPartition);

        // Create files
        File file1 = new File("Document1", 100, "/root/C:/Documents/Document1", "RW", "txt", "texttttt", new Date(),(int)Math.ceil((float)100/BLOCK_SIZE));
        File file2 = new File("Document2", 120, "/root/C:/Documents/Document2", "W", "txt", "another text", new Date(),(int)Math.ceil((float)120/BLOCK_SIZE));

        // Add files to folder
        documents.addChild(file1, rootPartition);
        documents.addChild(file2, rootPartition);

        File file3 = new File("mytext", 100, "/root/D:/mytext", "RWX", "txt", "hi :)", new Date(), (int)Math.ceil((float)100/BLOCK_SIZE));
        dDrive.addChild(file3, rootPartition);

        List<User> users = Arrays.asList(
                new User("mirna", new String[]{"CREATE", "READ", "WRITE", "DELETE"}),
                new User("verina", new String[]{"READ", "CREATE", "WRITE"}),
                new User("maria", new String[]{"READ", "WRITE"}),
                new User("marly", new String[]{"CREATE", "DELETE"})
        );

        // user input

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter your name: ");
        String userName = scanner.nextLine();

        User currentUser = findUserByName(userName, users);

        if (currentUser == null) {
            System.out.println("User not found. Exiting.");
            scanner.close();    // ma3le4 34an vscode 3ameli warning m3asabni
            return;
        }
        // Print welcome message
        System.out.println("Welcome, " + userName + "! You can perform the following operations on files: " + Arrays.toString(currentUser.allowedOperations));

        while(true) {

            System.out.println();
            System.out.println("What would you like to do (-1 to quit)?\n1. Navigate to file\n2. Navigate to folder\n3. Search for specific file");
            int choice = 0;
            
            try {
                choice = scanner.nextInt();
            }
            catch(Exception e) {
                scanner.next(); // Consume the invalid input
            }

            if(choice == -1) {
                System.out.println("Exiting..");
                break;
            }

            FileSystemObject selected = null;
            
            // 34an el scanner ahbal
            if(scanner.hasNextLine()){
                scanner.nextLine();
            }
            
            switch (choice) {
                case 1:
                    selected = rootPartition.navigateToFile(scanner);
                    break;

                case 2:
                    selected = rootPartition.navigateToFolder(scanner);
                    break;
                
                case 3:
                    selected = findFileByName(scanner, rootPartition);
                    break;

            
                default:
                    System.out.println("Invalid input. Please enter a number from 1-3.");
                    break;
            }

            if(selected == null) {
                continue;
            }

            System.out.println(selected.location);

            if(selected instanceof File) {
                showFileOptions(currentUser, selected, scanner, rootPartition);
            }
            else if(selected instanceof Folder) {
                showFolderOptions(currentUser, selected, scanner, rootPartition);
            }
            else {
                // m3raf4 eh el case di bs olt e7teyati m4 hane5sar 7aga
                System.out.println("Invalid selection.");
            }

            
        }
        // scanner.close();
    }
    

    private static User findUserByName(String name, List<User> users) {
        for (User user : users) {
            if (user.name.equalsIgnoreCase(name)) {
                return user;
            }
        }
        return null;
    }

    private static File findFileByName(Scanner scanner, Folder searchFolder) {

        // Prompt the user to enter the file name
        System.out.print("Enter the file name: ");
        String fileName = scanner.nextLine();

        // Search for files in the "searchFolder" folder with a matching name
        List<FileSystemObject> searchResults = searchFolder.searchDirectory(fileName);

        if (searchResults.isEmpty()) {
            System.out.println("File not found.");
        } else {
            // Display search results
            System.out.println("\nSearch Results:");
            for (int i = 0; i < searchResults.size(); i++) {
                FileSystemObject result = searchResults.get(i);
                System.out.println((i + 1) + ". " + result.name);
            }

            // Prompt the user to enter the file number
            System.out.print("Enter the index of the file you want to select: ");
            int fileNumber = scanner.nextInt();

            if (fileNumber >= 1 && fileNumber <= searchResults.size()) {
                FileSystemObject selectedFile = searchResults.get(fileNumber - 1);
                if (selectedFile instanceof File) {
                    return (File)selectedFile;
                } else {
                    System.out.println("\nSelected item is not a file.");
                }
            } else {
                System.out.println("\nInvalid file index.");
            }
        } 
        return null;
    }

    public static void showFileOptions(User currentUser, FileSystemObject selected, Scanner scanner, Partition rootPartition) {
        System.out.println("What would you like to with the selected file?\n1.Read\n2.View file information\n3.Write\n4.Move\n5.Rename\n6.Copy\n7.Delete\n8.Change Permission");
        int choice = 0;
        try {
            choice = scanner.nextInt();
        }
        catch(Exception e) {
            scanner.next(); // Consume the invalid input
        }

        // 34an el scanner ahbal
        if(scanner.hasNextLine()){
            scanner.nextLine();
        }
        if(choice == 1){
            // read
            if (currentUser.hasPermission("READ") && selected.protection.contains("R")) {
                System.out.println("\nReading file content: " + ((File)selected).content);
                ((File)selected).updateAccessTime();
            } else {
                System.out.println("\nUser does not have permission to read.");
            }
        } else if(choice == 2){
            // view info
            if (currentUser.hasPermission("READ") && selected.protection.contains("R")) {
                System.out.println("\nFile information:");
                System.out.println("Name: " + ((File)selected).name);
                System.out.println("ID: " + selected.identifier);
                System.out.println("Size: " + ((File)selected).size);
                System.out.println("Blocks: " + ((File)selected).blocks);
                System.out.println("Location: " + ((File)selected).location);
                System.out.println("Extension: " + ((File)selected).extension);
                System.out.println("Protection: " + ((File)selected).protection);
                System.out.println("Date created: " + ((File)selected).creationTime);
                System.out.println("Last modified: " + ((File)selected).modificationTime);
                System.out.println("Last accessed: " + ((File)selected).accessTime);
                ((File)selected).updateAccessTime();

            } else {
                System.out.println("\nUser does not have permission to read.");
            }
        } else if(choice == 3){
            // write
            if (currentUser.hasPermission("WRITE") && selected.protection.contains("W")) {
                // Prompt the user to enter the updated content
                System.out.print("Enter the updated content: ");
                String newContent = scanner.nextLine();
                ((File)selected).content = newContent;
                System.out.println("File content updated: " + ((File)selected).content);
                ((File)selected).updateAccessTime();
                ((File)selected).updateModificationTime();

            } else {
                System.out.println("\nUser does not have permission to write.");
            }
        } else if(choice == 4){
            // move
            if (currentUser.hasPermission("WRITE") && selected.protection.contains("W")) {

                System.out.println("Select new folder to move into: ");
                Folder newFolder = rootPartition.navigateToFolder(scanner);

                Folder parentFolder = selected.getParentFolder(rootPartition);
                int status = newFolder.addChild(selected, rootPartition);
                if(status != -1) {
                    System.out.println("File moved successfully.");
                    parentFolder.removeChild(selected.name);
                    selected.location = newFolder.location + "/" + selected.name;
                    ((File)selected).updateAccessTime();
                    ((File)selected).updateModificationTime();
                }
                

            } else {
                System.out.println("\nUser does not have permission to write.");
            }

        } else if(choice == 5){
            // rename
            if (currentUser.hasPermission("WRITE") && selected.protection.contains("W")) {
                
                System.out.print("Enter the new file name: ");
                String newName = scanner.nextLine();

                ((File)selected).renameFile(newName);
                ((File)selected).updateAccessTime();
                ((File)selected).updateModificationTime();

                System.out.println("File renamed successfully.");
                
            } else {
                System.out.println("\nUser does not have permission to write.");
            }

        } else if(choice == 6){
            // copy
            if (currentUser.hasPermission("WRITE") && selected.protection.contains("W")) {

                File newFile = ((File)selected).createCopy(rootPartition);
                ((File)selected).updateAccessTime();
                if(newFile != null) {
                    System.out.println("File copied at " + newFile.location);
                }
                
            } else {
                System.out.println("\nUser does not have permission to create.");
            }
        } else if(choice == 7){
            // delete
            if (currentUser.hasPermission("DELETE") && selected.protection.contains("X")) {

                Folder parent = selected.getParentFolder(rootPartition);
                parent.removeChild(selected.name);
                System.out.println("File deleted successfully.");
                
            } else {
                System.out.println("\nUser does not have permission to delete.");
            }

        } else if(choice == 8){
            // change permission
            if (currentUser.hasPermission("DELETE")) {

                System.out.println("Current protection: " + selected.protection);
                System.out.print("Enter new protection: ");
                String newProtection = scanner.nextLine();
                selected.updateProtection(newProtection);
                ((File)selected).updateAccessTime();
                ((File)selected).updateModificationTime();
                System.out.println("Protection updated successfully. New protection: " + selected.protection);
                
            } else {
                System.out.println("\nUser does not have permission to delete.");
            }
            
        } else {
            System.out.println("Invalid input.");
        }
    }

    public static void showFolderOptions(User currentUser, FileSystemObject selected, Scanner scanner, Partition rootPartition) {
        
        System.out.println("What would you like to with the selected file?\n1.Search\n2.View folder information\n3.Create new file\n4.Create new folder\n5.Delete");
        int choice = 0;
        try {
            choice = scanner.nextInt();
        }
        catch(Exception e) {
            scanner.next(); // Consume the invalid input
        }

        // 34an el scanner ahbal
        if(scanner.hasNextLine()){
            scanner.nextLine();
        }

        if(choice == 1){
            // search
            if (currentUser.hasPermission("READ")) {
                FileSystemObject selectedFile = findFileByName(scanner, (Folder)selected);
                showFileOptions(currentUser, selectedFile, scanner, rootPartition);

            } else {
                System.out.println("\nUser does not have permission to read.");
            }
            
        } else if(choice == 2){
            // view info
            if (currentUser.hasPermission("READ")) {
                System.out.println("\nFolder information:");
                System.out.println("Name: " + ((Folder)selected).name);
                System.out.println("ID: " + selected.name);                
                System.out.println("Size: " + ((Folder)selected).getUsedSize());
                if(selected instanceof Partition) {
                    System.out.println("Maximum Size: " + ((Partition)selected).maxSize);
                }
                System.out.println("Location: " + ((Folder)selected).location);
                System.out.println("Protection: " + ((Folder)selected).protection);
                
            } else {
                System.out.println("\nUser does not have permission to read.");
            }
        } else if(choice == 3){
            // create file
            if (currentUser.hasPermission("CREATE")) {
                ((Folder)selected).createFile(currentUser, scanner, rootPartition, BLOCK_SIZE);
            } else {
                System.out.println("\nUser does not have permission to create.");
            }
        } else if(choice == 4){
            // create folder
            if (currentUser.hasPermission("CREATE")) {
                ((Folder)selected).createFolder(currentUser, scanner, rootPartition);    
            } else {
                System.out.println("\nUser does not have permission to create.");
            }
        } else if(choice == 5){
            // delete
            if (currentUser.hasPermission("DELETE")) {

                Folder parent = selected.getParentFolder(rootPartition);
                parent.removeChild(selected.name);
                System.out.println("Folder deleted successfully.");
                
            } else {
                System.out.println("\nUser does not have permission to delete.");
            }
        } else {
            System.out.println("Invalid input.");
        }
    
    
    
    }

}
