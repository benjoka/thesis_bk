package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/aebruno/nwalgo"
	"github.com/gocarina/gocsv"
	"github.com/schollz/progressbar/v3"
	"io"
	"log"
	"math"
	"os"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type MutationSet struct { // Our example struct, you can use "-" to ignore a field
	Id             string `csv:"igs_id"`
	NCount         int    `csv:"totalMissing"`
	Substitutions  string `csv:"substitutions"`
	Insertions     string `csv:"insertions"`
	Deletions      string `csv:"deletions"`
	Missings       string `csv:"missing"`
	NonACGTNs      string `csv:"nonACGTNs"`
	AlignmentStart int    `csv:"alignmentStart"`
	AlignmentEnd   int    `csv:"alignmentEnd"`
	Candidates     []int  `csv:"candidates"`
}

var referenceSequence string
var ambiguousCharacters = map[string][]string{"A": {"A"},
	"C": {"C"},
	"G": {"G"},
	"T": {"T"},
	"U": {"U"},
	"M": {"A", "C"},
	"R": {"A", "G"},
	"S": {"C", "G"},
	"W": {"A", "T"},
	"Y": {"C", "T"},
	"K": {"G", "T"},
	"V": {"A", "C", "G"},
	"H": {"A", "C", "T"},
	"D": {"A", "G", "T"},
	"B": {"C", "G", "T"},
	"X": {"A", "C", "G", "T"}}

func main() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter process name: ")
	processName, _ := reader.ReadString('\n')
	fmt.Print("Enter sequence file path: ")
	sequencesFilePath, _ := reader.ReadString('\n')
	var useCandidates bool
	flag.BoolVar(&useCandidates, "candidates", false, "Use similarity candidates")
	var fast bool
	flag.BoolVar(&fast, "fast", false, "Use optimization")
	flag.Parse()
	referenceSequence = readFromTxt("reference.txt")
	positionMutationMap, identifierMapping, nCountMapping, candidates, mutationSets := extractDataFromCsv(sequencesFilePath)
	start := time.Now()
	distanceMap := make(map[int]map[int]float64)
	var wg sync.WaitGroup
	var mutex sync.Mutex
	bar := progressbar.Default(int64((len(positionMutationMap) * (len(positionMutationMap) - 1)) / 2))
	for i := 0; i < len(identifierMapping); i++ {
		distanceMap[i] = make(map[int]float64)
		distanceMap[i][i] = 0
		for j := 0; j < i; j++ {
			wg.Add(1)
			bar.Add(1)
			go func(i int, j int) {
				mutex.Lock()
				distanceMap[i][j] = math.Inf(1)
				mutex.Unlock()
				if !useCandidates || slices.Contains(candidates[i], j) {
					sequence1, sequence2 := alignSequences(positionMutationMap[i], positionMutationMap[j], mutationSets[i], mutationSets[j], fast)
					distance := calculateDistanceBetweenSequences(sequence1, sequence2, nCountMapping[i], nCountMapping[j], fast)
					mutex.Lock()
					distanceMap[i][j] = distance
					mutex.Unlock()
				}
				wg.Done()
			}(i, j)
		}
		wg.Wait()
	}
	elapsed := time.Since(start)
	fmt.Printf("Time elapsed: %s\n", elapsed)
	finishProcess(distanceMap, identifierMapping, processName, useCandidates)
}

func finishProcess(distanceMap map[int]map[int]float64, identifierMapping map[int]string, processName string, useCandidates bool) {
	distanceMatrix := assembleDistanceMatrix(distanceMap)
	sortedDistanceMatrix := sortDistanceMatrix(distanceMatrix)
	processName = strings.TrimSpace(processName)

	if useCandidates {
		persistDistanceMatrixAsJson(distanceMap, processName)
	} else {
		persistDistanceMatrixAsCsv(sortedDistanceMatrix, identifierMapping, processName)
	}

	fmt.Println("Distances matrix was persisted as distance_matrix_" + processName + ".csv")
}

func extractDataFromCsv(fileName string) (map[int]map[int]map[string]string, map[int]string, map[int]int, map[int][]int, []MutationSet) {
	mutationSets := readCsvFile(fileName)
	positionMutations := make(map[int]map[int]map[string]string)
	identifierMapping := make(map[int]string)
	nCountMapping := make(map[int]int)
	candidates := make(map[int][]int)
	for i := 0; i < len(mutationSets); i++ {
		identifierMapping[i] = mutationSets[i].Id
		nCountMapping[i] = mutationSets[i].NCount
		positionMutations[i] = getPositionMutationsForSequence(mutationSets[i])
		candidates[i] = mutationSets[i].Candidates
	}
	return positionMutations, identifierMapping, nCountMapping, candidates, mutationSets
}

func readCsvFile(filePath string) []MutationSet {
	sequencesFile, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer sequencesFile.Close()

	mutationSets := []MutationSet{}

	gocsv.SetCSVReader(func(in io.Reader) gocsv.CSVReader {
		r := csv.NewReader(in)
		r.Comma = ';'
		return r
	})

	if err := gocsv.UnmarshalFile(sequencesFile, &mutationSets); err != nil {
		panic(err)
	}
	if _, err := sequencesFile.Seek(0, 0); err != nil { // Go to the start of the file
		panic(err)
	}

	sort.Slice(mutationSets[:], func(i, j int) bool {
		return mutationSets[i].Id < mutationSets[j].Id
	})

	return mutationSets
}

func readFromTxt(filePath string) string {
	file, _ := os.Open(filePath)
	defer file.Close()
	r := bufio.NewReader(file)
	// Section 2
	var buffer bytes.Buffer
	for {
		line, _, err := r.ReadLine()
		if len(line) > 0 {
			buffer.WriteString(string(line))
		}
		if err != nil {
			break
		}
	}
	return buffer.String()
}

func getPositionMutationsForSequence(mutationSet MutationSet) map[int]map[string]string {
	positionMutations := make(map[int]map[string]string)
	convertMutationsStringToArray(mutationSet.Missings)
	positionMutations = extractPositionSubstitutions(convertMutationsStringToArray(mutationSet.Substitutions), positionMutations)
	positionMutations = extractPositionInsertions(convertMutationsStringToArray(mutationSet.Insertions), positionMutations)
	positionMutations = extractPositionDeletions(convertMutationsStringToArray(mutationSet.Deletions), positionMutations)
	positionMutations = extractPositionMissings(convertMutationsStringToArray(mutationSet.Missings), positionMutations)
	positionMutations = extractPositionNonACGTNs(convertMutationsStringToArray(mutationSet.NonACGTNs), positionMutations)
	positionMutations = extractPositionPaddings(mutationSet.AlignmentStart, mutationSet.AlignmentEnd, positionMutations)
	return positionMutations
}

func convertMutationsStringToArray(mutationsString string) []string {
	if mutationsString == "" {
		return nil
	}
	mutationsArray := strings.Split(mutationsString, ",")
	return mutationsArray
}

func extractPositionPaddings(alignmentStart int, alignmentEnd int, positionMutations map[int]map[string]string) map[int]map[string]string {
	// csv mutation string indices start at 1
	alignmentStart--
	for position := 0; position < alignmentStart; position++ {
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["del"] = "-"
	}
	for position := alignmentEnd; position < len(referenceSequence); position++ {
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["del"] = "-"
	}
	return positionMutations
}

func extractPositionSubstitutions(substitutions []string, positionMutations map[int]map[string]string) map[int]map[string]string {
	for _, substitutionString := range substitutions {
		positionMutations = addSubstitution(substitutionString, positionMutations)
	}
	return positionMutations
}

func addSubstitution(substitutionString string, positionMutations map[int]map[string]string) map[int]map[string]string {
	re := regexp.MustCompile(`^([A-Za-z])(\d+)([A-Za-z])$`)
	matches := re.FindStringSubmatch(substitutionString)
	position, _ := strconv.Atoi(matches[2])
	// csv mutation string indices start at 1
	position--
	character := matches[3]
	if positionMutations[position] == nil {
		positionMutations[position] = make(map[string]string)
	}
	positionMutations[position]["snp"] = character
	return positionMutations
}

func extractPositionInsertions(insertions []string, positionMutations map[int]map[string]string) map[int]map[string]string {
	for _, insertionString := range insertions {
		positionMutations = addInsertion(insertionString, positionMutations)
	}
	return positionMutations
}

func addInsertion(insertionString string, positionMutations map[int]map[string]string) map[int]map[string]string {
	re := regexp.MustCompile(`^(\d+):(.+)$`)
	matches := re.FindStringSubmatch(insertionString)
	position, _ := strconv.Atoi(matches[1])
	// csv mutation string indices start at 1
	position--
	character := matches[2]
	if positionMutations[position] == nil {
		positionMutations[position] = make(map[string]string)
	}
	positionMutations[position]["ins"] = character
	return positionMutations
}

func extractPositionDeletions(deletions []string, positionMutations map[int]map[string]string) map[int]map[string]string {
	for _, deletionString := range deletions {
		positionMutations = addDeletion(deletionString, positionMutations)
	}
	return positionMutations
}

func addDeletion(deletionString string, positionMutations map[int]map[string]string) map[int]map[string]string {
	re := regexp.MustCompile(`^(\d+)-(\d+)$`)
	matches := re.FindStringSubmatch(deletionString)
	if len(matches) == 0 {
		position, _ := strconv.Atoi(deletionString)
		position--
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["del"] = "-"
		return positionMutations
	}
	begin, _ := strconv.Atoi(matches[1])
	// csv mutation string indices start at 1
	begin--
	end, _ := strconv.Atoi(matches[2])
	for position := begin; position < end; position++ {
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["del"] = "-"
	}
	return positionMutations
}

func extractPositionMissings(missings []string, positionMutations map[int]map[string]string) map[int]map[string]string {
	for _, missingString := range missings {
		positionMutations = addMissing(missingString, positionMutations)
	}
	return positionMutations
}

func addMissing(missingString string, positionMutations map[int]map[string]string) map[int]map[string]string {
	re := regexp.MustCompile(`^(\d+)-(\d+)$`)
	matches := re.FindStringSubmatch(missingString)
	if len(matches) == 0 {
		position, _ := strconv.Atoi(missingString)
		position--
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["snp"] = "N"
		return positionMutations
	}
	begin, _ := strconv.Atoi(matches[1])
	// csv mutation string indices start at 1
	begin--
	end, _ := strconv.Atoi(matches[2])
	for position := begin; position < end; position++ {
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["snp"] = "N"
	}
	return positionMutations
}

func extractPositionNonACGTNs(nonACGTnStrings []string, positionMutations map[int]map[string]string) map[int]map[string]string {
	for _, nonACGTnString := range nonACGTnStrings {
		positionMutations = addNonACGTNs(nonACGTnString, positionMutations)
	}
	return positionMutations
}

// not tested yet
func addNonACGTNs(nonACGTnString string, positionMutations map[int]map[string]string) map[int]map[string]string {
	re := regexp.MustCompile(`(.+):(\d+)(?:-(\d+))?`)
	matches := re.FindStringSubmatch(nonACGTnString)
	character := matches[1]
	// csv mutation string indices start at 1
	begin, _ := strconv.Atoi(matches[2])
	begin--
	end := begin + 1
	if len(matches) == 2 {
		end, _ = strconv.Atoi(matches[3])
	}
	for position := begin; position < end; position++ {
		if positionMutations[position] == nil {
			positionMutations[position] = make(map[string]string)
		}
		positionMutations[position]["snp"] = character
	}
	return positionMutations

}

func alignSequences(mutationsMapSample1 map[int]map[string]string, mutationsMapSample2 map[int]map[string]string, mutationSet1 MutationSet, mutationSet2 MutationSet, fast bool) ([]string, []string) {
	var sequence1 []string
	var sequence2 []string

	properCharsThreshold := 20
	properCharsSeen1 := 0
	properCharsSeen2 := 0
	properCharsAmount1 := mutationSet1.AlignmentEnd - mutationSet1.AlignmentStart - mutationSet1.NCount
	properCharsAmount2 := mutationSet2.AlignmentEnd - mutationSet2.AlignmentStart - mutationSet2.NCount

	for referenceIndex, referenceValue := range referenceSequence {
		referenceNucleotide := string(referenceValue)
		var additions1 []string
		var additions2 []string
		mutations1, mutationsExist1 := mutationsMapSample1[referenceIndex]
		mutations2, mutationsExist2 := mutationsMapSample2[referenceIndex]
		substitutionCharacter1, substitutionExists1 := mutations1["snp"]
		substitutionCharacter2, substitutionExists2 := mutations2["snp"]
		deletionCharacter1, deletionExists1 := mutations1["del"]
		deletionCharacter2, deletionExists2 := mutations2["del"]
		insertionCharacter1, insertionExists1 := mutations1["ins"]
		insertionCharacter2, insertionExists2 := mutations2["ins"]

		if fast {
			if !deletionExists1 && !(!insertionExists1 && insertionExists2) {
				properCharsSeen1++
			}
			if !deletionExists2 && !(insertionExists1 && !insertionExists2) {
				properCharsSeen2++
			}
			if properCharsReached(properCharsSeen1, properCharsSeen1, properCharsAmount1, properCharsAmount2, properCharsThreshold) {
				continue
			}
			if !mutationsExist1 && !mutationsExist2 {
				continue
			}
		}

		// add reference chars to additions in case not mutations exist for the current reference position
		if !mutationsExist1 {
			additions1 = append(additions1, referenceNucleotide)
		}
		if !mutationsExist2 {
			additions2 = append(additions2, referenceNucleotide)
		}

		// handle substitutions
		if fast {
			if substitutionExists1 && substitutionCharacter1 == "N" || substitutionExists2 && substitutionCharacter2 == "N" {
				continue
			}
		}
		if substitutionExists1 {
			additions1 = append(additions1, substitutionCharacter1)
		}
		if substitutionExists2 {
			additions2 = append(additions2, substitutionCharacter2)
		}

		// handle deletions
		if deletionExists1 && !deletionExists2 {
			additions1 = append(additions1, deletionCharacter1)
		}
		if deletionExists2 && !deletionExists1 {
			additions2 = append(additions2, deletionCharacter2)
		}

		// handle insertions
		if insertionExists1 && insertionExists2 && insertionCharacter1 != insertionCharacter2 {
			alignment1, alignment2, _ := nwalgo.Align(insertionCharacter1, insertionCharacter2, 1, -1, -1)
			additions1 = append(additions1, strings.Split(alignment1, "")...)
			additions2 = append(additions2, strings.Split(alignment2, "")...)
			if len(mutations1) > 1 {
				additions1 = append(additions1, strings.Split(alignment1, "")...)
			} else {
				additions1 = append([]string{referenceNucleotide}, additions1...)
				additions1 = append(additions1, strings.Split(alignment1, "")...)
			}
			if len(mutations2) > 1 {
				additions2 = append(additions2, strings.Split(alignment2, "")...)
			} else {
				additions2 = append([]string{referenceNucleotide}, additions2...)
				additions2 = append(additions2, strings.Split(alignment2, "")...)
			}
		}
		if insertionExists1 && insertionExists2 && insertionCharacter1 == insertionCharacter2 {
			if len(mutations1) > 1 {
				additions1 = append(additions1, strings.Split(insertionCharacter1, "")...)
			} else {
				additions1 = append([]string{referenceNucleotide}, additions1...)
				additions1 = append(additions1, strings.Split(insertionCharacter1, "")...)
			}
			if len(mutations1) > 1 {
				additions2 = append(additions2, strings.Split(insertionCharacter2, "")...)
			} else {
				additions2 = append([]string{referenceNucleotide}, additions2...)
				additions2 = append(additions2, strings.Split(insertionCharacter2, "")...)
			}
		} else if insertionExists1 && !insertionExists2 {
			insertionLength := len(insertionCharacter1)
			if len(mutations1) > 1 {
				additions1 = append(additions1, strings.Split(insertionCharacter1, "")...)
			} else {
				additions1 = append([]string{referenceNucleotide}, additions1...)
				additions1 = append(additions1, strings.Split(insertionCharacter1, "")...)
			}
			additions2 = append(additions2, strings.Split(strings.Repeat("-,", insertionLength), ",")[:insertionLength]...)
		} else if insertionExists2 && !insertionExists1 {
			insertionLength := len(insertionCharacter2)
			if len(mutations2) > 1 {
				additions2 = append(additions2, strings.Split(insertionCharacter2, "")...)
			} else {
				additions2 = append([]string{referenceNucleotide}, additions2...)
				additions2 = append(additions2, strings.Split(insertionCharacter2, "")...)
			}
			additions1 = append(additions1, strings.Split(strings.Repeat("-,", insertionLength), ",")[:insertionLength]...)
		}

		sequence1 = append(sequence1, additions1...)
		sequence2 = append(sequence2, additions2...)
	}
	return sequence1, sequence2
}

func calculateDistanceBetweenSequences(sequence1 []string, sequence2 []string, nCount1 int, nCount2 int, fast bool) float64 {
	distance := 0.0
	sequenceLength1 := len(sequence1)
	sequenceLength2 := len(sequence2)
	activeGap1 := false
	activeGap2 := false
	properCharsThreshold := 20
	properCharsSeen1 := 0
	properCharsSeen2 := 0

	sequenceLength := int(math.Min(float64(len(sequence1)), float64(len(sequence2))))

	for i := 0; i < sequenceLength; i++ {
		character1 := sequence1[i]
		character2 := sequence2[i]

		if character1 == "N" || character2 == "N" {
			continue
		}

		if !fast {
			if character1 != "-" {
				properCharsSeen1++
			}
			if character2 != "-" {
				properCharsSeen2++
			}
			properCharsAmount1 := sequenceLength1 - nCount1
			properCharsAmount2 := sequenceLength2 - nCount2
			if properCharsReached(properCharsSeen1, properCharsSeen1, properCharsAmount1, properCharsAmount2, properCharsThreshold) {
				continue
			}
		}

		if character1 == character2 {
			continue
		}

		if character1 == "-" {
			if !activeGap1 {
				distance++
			}
			activeGap1 = true
			activeGap2 = false
		}
		if character2 == "-" {
			if !activeGap2 {
				distance++
			}
			activeGap1 = false
			activeGap2 = true
		}
		if character1 != "-" && character2 != "-" && !slices.Contains(ambiguousCharacters[character1], character2) &&
			!slices.Contains(ambiguousCharacters[character2], character1) {
			distance++
			activeGap1 = false
			activeGap2 = false
		}
	}

	return distance
}

func assembleDistanceMatrix(distanceMap map[int]map[int]float64) map[int]map[int]float64 {
	distanceMatrix := make(map[int]map[int]float64)
	for i := 0; i < len(distanceMap); i++ {
		if distanceMatrix[i] == nil {
			distanceMatrix[i] = make(map[int]float64)
		}
		for j := 0; j <= i; j++ {
			distance := distanceMap[i][j]
			distanceMatrix[i][j] = distance
			if distanceMatrix[j] == nil {
				distanceMatrix[j] = make(map[int]float64)
			}
			distanceMatrix[j][i] = distance

		}
	}
	return distanceMatrix
}

func sortDistanceMatrix(distanceMatrix map[int]map[int]float64) map[int]map[int]float64 {
	keys := make([]int, 0, len(distanceMatrix))
	for k := range distanceMatrix {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	sortedDistanceMatrix := make(map[int]map[int]float64)
	for _, k := range keys {
		sortedDistanceMatrix[k] = distanceMatrix[k]
	}
	return sortedDistanceMatrix
}

func persistDistanceMatrixAsCsv(distanceMatrix map[int]map[int]float64, identifierMapping map[int]string, processName string) {
	file, err := os.Create("distance_matrix_" + processName + ".csv")
	defer file.Close()
	if err != nil {
		log.Fatalln("failed to open file", err)
	}
	w := csv.NewWriter(file)
	w.Comma = ';'
	defer w.Flush()
	var data [][]string

	headerRow := []string{""}
	distanceRows := [][]string{}
	for i := 0; i < len(distanceMatrix); i++ {
		headerRow = append(headerRow, identifierMapping[i])
		distances := distanceMatrix[i]
		row := []string{identifierMapping[i]}
		for j := 0; j < len(distanceMatrix); j++ {
			distance := distances[j]
			if distance == math.Inf(1) {
				row = append(row, "")
			} else {
				row = append(row, strconv.Itoa(int(distance)))
			}
		}
		distanceRows = append(distanceRows, row)
	}
	data = append(data, headerRow)
	data = append(data, distanceRows...)
	w.WriteAll(data)
}

func persistDistanceMatrixAsJson(distanceMap map[int]map[int]float64, processName string) {
	for i := 0; i < len(distanceMap); i++ {
		for j := 0; j <= i; j++ {
			distance := distanceMap[i][j]
			if distance == math.Inf(1) {
				delete(distanceMap[i], j)
			}
		}
	}
	jsonData, err := json.Marshal(distanceMap)
	if err != nil {
		panic(err)
	}
	// Write to a file
	err = os.WriteFile("distances_"+processName+".json", jsonData, 0644)
	if err != nil {
		panic(err)
	}
}

func properCharsReached(properCharsSeen1 int, properCharsSeen2 int, properCharsAmount1 int, properCharsAmount2 int, properCharsThreshold int) bool {
	return properCharsSeen1 < properCharsThreshold || properCharsSeen2 < properCharsThreshold ||
		(properCharsAmount1-properCharsSeen1 < properCharsThreshold) || (properCharsAmount2-properCharsSeen2 < properCharsThreshold)
}
