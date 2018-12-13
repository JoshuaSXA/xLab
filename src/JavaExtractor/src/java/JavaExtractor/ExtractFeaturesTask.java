package JavaExtractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Callable;

import com.github.javaparser.ast.Node;
import org.apache.commons.lang3.StringUtils;

import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.ProgramFeatures;

import javax.swing.text.StyledEditorKit;

public class ExtractFeaturesTask implements Callable<Void> {
	HashMap<Node, ArrayList<Node>> tree;

	CommandLineValues m_CommandLineValues;
	CompilationUnit m_CompilationUnit;
	String code = null;
	Path filePath;

	public ExtractFeaturesTask(CommandLineValues commandLineValues, Path path) {
		m_CommandLineValues = commandLineValues;
		this.filePath = path;
		try {
			this.code = new String(Files.readAllBytes(path));
		} catch (IOException e) {
			e.printStackTrace();
			this.code = Common.EmptyString;
		}
	}

	@Override
	public Void call() throws Exception {
		//System.err.println("Extracting file: " + filePath);
		processFile();
		//System.err.println("Done with file: " + filePath);
		return null;
	}

	public void processFile() {
		ArrayList<ProgramFeatures> features;
		try {
			features = extractSingleFile();
		} catch (ParseException | IOException e) {
			e.printStackTrace();
			return;
		}
		if (features == null) {
			return;
		}
		
		String toPrint = featuresToString(features);
		if (toPrint.length() > 0) {
			System.out.println(toPrint.replaceAll(" ", "\n"));
		}

		System.out.println("\nTree Structure Starts Here.\n");

        Iterator iter = this.tree.entrySet().iterator();
		while (iter.hasNext()) {
		    HashMap.Entry<Node, ArrayList<Node>> entry = (HashMap.Entry<Node, ArrayList<Node>>) iter.next();
		    Node key = (Node) entry.getKey();
		    ArrayList<Node> arr = (ArrayList<Node>) (entry.getValue());
		    StringBuffer sb = new StringBuffer();
		    for (Node node : arr) {
		        sb.append(System.identityHashCode(node));
		        sb.append("|");
            }
            System.out.printf("Parent: %d, Child: %s\n", System.identityHashCode(key), sb.toString());
        }
	}

	public ArrayList<ProgramFeatures> extractSingleFile() throws ParseException, IOException {
		FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues, code);

		ArrayList<ProgramFeatures> features = featureExtractor.extractFeatures();

		m_CompilationUnit = featureExtractor.getParsedFile();

        this.tree = featureExtractor.getTree();

		return features;
	}

	public String featuresToString(ArrayList<ProgramFeatures> features) {
		if (features == null || features.isEmpty()) {
			return Common.EmptyString;
		}

		List<String> methodsOutputs = new ArrayList<>();

		for (ProgramFeatures singleMethodfeatures : features) {
			StringBuilder builder = new StringBuilder();
			
			String toPrint = Common.EmptyString;
			toPrint = singleMethodfeatures.toString();
			if (m_CommandLineValues.PrettyPrint) {
				toPrint = toPrint.replace(" ", "\n\t");
			}
			builder.append(toPrint);

			methodsOutputs.add(builder.toString());

		}
		return StringUtils.join(methodsOutputs, "\n");
	}
}
