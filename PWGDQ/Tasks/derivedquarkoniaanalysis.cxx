// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// V0 analysis task
// ================
//
// This code loops over a V0Cores table and produces some
// standard analysis output. It is meant to be run over
// derived data.
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    romain.schotter@cern.ch
//    david.dobrigkeit.chinellato@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/O2DatabasePDGPlugin.h"
#include "ReconstructionDataFormats/Track.h"
#include "CommonConstants/PhysicsConstants.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Multiplicity.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "PWGUD/Core/SGSelector.h"
#include "Tools/ML/MlResponse.h"
#include "Tools/ML/model.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

using dauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;
using dauMCTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackMCIds, aod::DauTrackTPCPIDs>;
using v0Candidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0LambdaMLScores, aod::V0AntiLambdaMLScores, aod::V0K0ShortMLScores>;
// using v0MCCandidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0MCCores, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0MCMothers, aod::V0MCCollRefs>;
using v0MCCandidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0MCMothers, aod::V0CoreMCLabels, aod::V0LambdaMLScores, aod::V0AntiLambdaMLScores, aod::V0K0ShortMLScores>;

// simple checkers, but ensure 64 bit integers
#define bitset(var, nbit) ((var) |= (static_cast<uint64_t>(1) << static_cast<uint64_t>(nbit)))
#define bitcheck(var, nbit) ((var) & (static_cast<uint64_t>(1) << static_cast<uint64_t>(nbit)))

struct derivedquarkoniaanalysis {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // master analysis switches
  Configurable<bool> isPP{"isPP", true, "If running on pp collision, switch it on true"};
  Configurable<bool> calculateFeeddownMatrix{"calculateFeeddownMatrix", true, "fill Lambda feeddown matrix if MC"};

  Configurable<bool> requireSel8{"requireSel8", true, "require sel8 event selection"};
  Configurable<bool> rejectITSROFBorder{"rejectITSROFBorder", true, "reject events at ITS ROF border"};
  Configurable<bool> rejectTFBorder{"rejectTFBorder", true, "reject events at TF border"};
  Configurable<bool> requireIsVertexITSTPC{"requireIsVertexITSTPC", false, "require events with at least one ITS-TPC track"};
  Configurable<bool> requireIsGoodZvtxFT0VsPV{"requireIsGoodZvtxFT0VsPV", true, "require events with PV position along z consistent (within 1 cm) between PV reconstructed using tracks and PV using FT0 A-C time difference"};
  Configurable<bool> requireIsVertexTOFmatched{"requireIsVertexTOFmatched", false, "require events with at least one of vertex contributors matched to TOF"};
  Configurable<bool> requireIsVertexTRDmatched{"requireIsVertexTRDmatched", false, "require events with at least one of vertex contributors matched to TRD"};
  Configurable<bool> rejectSameBunchPileup{"rejectSameBunchPileup", true, "reject collisions in case of pileup with another collision in the same foundBC"};
  Configurable<bool> requireNoCollInTimeRangeStd{"requireNoCollInTimeRangeStd", true, "reject collisions corrupted by the cannibalism, with other collisions within +/- 10 microseconds"};
  Configurable<bool> requireNoCollInTimeRangeNarrow{"requireNoCollInTimeRangeNarrow", false, "reject collisions corrupted by the cannibalism, with other collisions within +/- 10 microseconds"};

  // fast check on occupancy
  Configurable<float> minOccupancy{"minOccupancy", -1, "minimum occupancy from neighbouring collisions"};
  Configurable<float> maxOccupancy{"maxOccupancy", -1, "maximum occupancy from neighbouring collisions"};

  struct : ConfigurableGroup {
    Configurable<int> v0TypeSelection{"v0Selections.v0TypeSelection", 1, "select on a certain V0 type (leave negative if no selection desired)"};

    // Selection criteria: acceptance
    Configurable<float> rapidityCut{"v0Selections.rapidityCut", 0.5, "rapidity"};
    Configurable<float> daughterEtaCut{"v0Selections.daughterEtaCut", 0.8, "max eta for daughters"};

    // Standard 5 topological criteria
    Configurable<float> v0cospa{"v0Selections.v0cospa", 0.97, "min V0 CosPA"};
    Configurable<float> dcav0dau{"v0Selections.dcav0dau", 1.0, "max DCA V0 Daughters (cm)"};
    Configurable<float> dcav0topv{"v0Selections.dcav0topv", .05, "min DCA V0 to PV (cm)"};
    Configurable<float> dcapiontopv{"v0Selections.dcapiontopv", .05, "min DCA Pion To PV (cm)"};
    Configurable<float> dcaprotontopv{"v0Selections.dcaprotontopv", .05, "min DCA Proton To PV (cm)"};
    Configurable<float> v0radius{"v0Selections.v0radius", 1.2, "minimum V0 radius (cm)"};
    Configurable<float> v0radiusMax{"v0Selections.v0radiusMax", 1E5, "maximum V0 radius (cm)"};

    // invariant mass selection
    Configurable<float> v0MassWindow{"v0Selections.v0MassWindow", 0.008, "#Lambda mass (GeV/#it{c}^{2})"};

    // Additional selection on the AP plot (exclusive for K0Short)
    // original equation: lArmPt*5>TMath::Abs(lArmAlpha)
    Configurable<float> armPodCut{"v0Selections.armPodCut", 5.0f, "pT * (cut) > |alpha|, AP cut. Negative: no cut"};

    // Track quality
    Configurable<int> minTPCrows{"v0Selections.minTPCrows", 70, "minimum TPC crossed rows"};
    Configurable<int> minITSclusters{"v0Selections.minITSclusters", -1, "minimum ITS clusters"};
    Configurable<bool> skipTPConly{"v0Selections.skipTPConly", false, "skip V0s comprised of at least one TPC only prong"};
    Configurable<bool> requirePosITSonly{"v0Selections.requirePosITSonly", false, "require that positive track is ITSonly (overrides TPC quality)"};
    Configurable<bool> requireNegITSonly{"v0Selections.requireNegITSonly", false, "require that negative track is ITSonly (overrides TPC quality)"};

    // PID (TPC/TOF)
    Configurable<float> TpcPidNsigmaCut{"v0Selections.TpcPidNsigmaCut", 5, "TpcPidNsigmaCut"};
    Configurable<float> TofPidNsigmaCutLaPr{"v0Selections.TofPidNsigmaCutLaPr", 1e+6, "TofPidNsigmaCutLaPr"};
    Configurable<float> TofPidNsigmaCutLaPi{"v0Selections.TofPidNsigmaCutLaPi", 1e+6, "TofPidNsigmaCutLaPi"};
    Configurable<float> TofPidNsigmaCutK0Pi{"v0Selections.TofPidNsigmaCutK0Pi", 1e+6, "TofPidNsigmaCutK0Pi"};

    // PID (TOF)
    Configurable<float> maxDeltaTimeProton{"v0Selections.maxDeltaTimeProton", 1e+9, "check maximum allowed time"};
    Configurable<float> maxDeltaTimePion{"v0Selections.maxDeltaTimePion", 1e+9, "check maximum allowed time"};
  } v0Selections;

  Configurable<bool> doCompleteTopoQA{"doCompleteTopoQA", false, "do topological variable QA histograms"};
  Configurable<bool> doTPCQA{"doTPCQA", false, "do TPC QA histograms"};
  Configurable<bool> doTOFQA{"doTOFQA", false, "do TOF QA histograms"};
  Configurable<int> doDetectPropQA{"doDetectPropQA", 0, "do Detector/ITS map QA: 0: no, 1: 4D, 2: 5D with mass"};

  Configurable<bool> doPlainTopoQA{"doPlainTopoQA", true, "do simple 1D QA of candidates"};
  Configurable<float> qaMinPt{"qaMinPt", 0.0f, "minimum pT for QA plots"};
  Configurable<float> qaMaxPt{"qaMaxPt", 1000.0f, "maximum pT for QA plots"};
  Configurable<bool> qaCentrality{"qaCentrality", false, "qa centrality flag: check base raw values"};

  // rapidity cut for the quarkonium mother
  Configurable<float> rapidityQuarkoniumCut{"rapidityQuarkoniumCut", 1e+9, "rapidity of quarkonium particle"};
  
  // for MC
  Configurable<bool> doMCAssociation{"doMCAssociation", true, "if MC, do MC association"};
  Configurable<bool> doCollisionAssociationQA{"doCollisionAssociationQA", true, "check collision association"};

  // Machine learning evaluation for pre-selection and corresponding information generation
  o2::ml::OnnxModel mlCustomModelK0Short;
  o2::ml::OnnxModel mlCustomModelLambda;
  o2::ml::OnnxModel mlCustomModelAntiLambda;
  o2::ml::OnnxModel mlCustomModelGamma;

  struct : ConfigurableGroup {
    // ML classifiers: master flags to control whether we should use custom ML classifiers or the scores in the derived data
    Configurable<bool> useK0ShortScores{"mlConfigurations.useK0ShortScores", false, "use ML scores to select K0Short"};
    Configurable<bool> useLambdaScores{"mlConfigurations.useLambdaScores", false, "use ML scores to select Lambda"};
    Configurable<bool> useAntiLambdaScores{"mlConfigurations.useAntiLambdaScores", false, "use ML scores to select AntiLambda"};

    Configurable<bool> calculateK0ShortScores{"mlConfigurations.calculateK0ShortScores", false, "calculate K0Short ML scores"};
    Configurable<bool> calculateLambdaScores{"mlConfigurations.calculateLambdaScores", false, "calculate Lambda ML scores"};
    Configurable<bool> calculateAntiLambdaScores{"mlConfigurations.calculateAntiLambdaScores", false, "calculate AntiLambda ML scores"};

    // ML input for ML calculation
    Configurable<std::string> customModelPathCCDB{"mlConfigurations.customModelPathCCDB", "", "Custom ML Model path in CCDB"};
    Configurable<int64_t> timestampCCDB{"mlConfigurations.timestampCCDB", -1, "timestamp of the ONNX file for ML model used to query in CCDB.  Exceptions: > 0 for the specific timestamp, 0 gets the run dependent timestamp"};
    Configurable<bool> loadCustomModelsFromCCDB{"mlConfigurations.loadCustomModelsFromCCDB", false, "Flag to enable or disable the loading of custom models from CCDB"};
    Configurable<bool> enableOptimizations{"mlConfigurations.enableOptimizations", false, "Enables the ONNX extended model-optimization: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED)"};

    // Local paths for test purposes
    Configurable<std::string> localModelPathLambda{"mlConfigurations.localModelPathLambda", "Lambda_BDTModel.onnx", "(std::string) Path to the local .onnx file."};
    Configurable<std::string> localModelPathAntiLambda{"mlConfigurations.localModelPathAntiLambda", "AntiLambda_BDTModel.onnx", "(std::string) Path to the local .onnx file."};
    Configurable<std::string> localModelPathK0Short{"mlConfigurations.localModelPathK0Short", "KZeroShort_BDTModel.onnx", "(std::string) Path to the local .onnx file."};

    // Thresholds for choosing to populate V0Cores tables with pre-selections
    Configurable<float> thresholdLambda{"mlConfigurations.thresholdLambda", -1.0f, "Threshold to keep Lambda candidates"};
    Configurable<float> thresholdAntiLambda{"mlConfigurations.thresholdAntiLambda", -1.0f, "Threshold to keep AntiLambda candidates"};
    Configurable<float> thresholdK0Short{"mlConfigurations.thresholdK0Short", -1.0f, "Threshold to keep K0Short candidates"};
  } mlConfigurations;

  // CCDB options
  struct : ConfigurableGroup {
    Configurable<std::string> ccdburl{"ccdbConfigurations.ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
    Configurable<std::string> grpPath{"ccdbConfigurations.grpPath", "GLO/GRP/GRP", "Path of the grp file"};
    Configurable<std::string> grpmagPath{"ccdbConfigurations.grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
    Configurable<std::string> lutPath{"ccdbConfigurations.lutPath", "GLO/Param/MatLUT", "Path of the Lut parametrization"};
    Configurable<std::string> geoPath{"ccdbConfigurations.geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
    Configurable<std::string> mVtxPath{"ccdbConfigurations.mVtxPath", "GLO/Calib/MeanVertex", "Path of the mean vertex file"};
  } ccdbConfigurations;

  o2::ccdb::CcdbApi ccdbApi;
  int mRunNumber;
  std::map<std::string, std::string> metadata;

  static constexpr float defaultLifetimeCuts[1][2] = {{30., 20.}};
  Configurable<LabeledArray<float>> lifetimecut{"lifetimecut", {defaultLifetimeCuts[0], 2, {"lifetimecutLambda", "lifetimecutK0S"}}, "lifetimecut"};

  ConfigurableAxis axisV0Pt{"axisV0Pt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for V0s analysis"};
  ConfigurableAxis axisCascPt{"axisCascPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for feeddown from Xi"};
  ConfigurableAxis axisV0PtCoarse{"axisV0PtCoarse", {VARIABLE_WIDTH, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 10.0f, 15.0f}, "pt axis for QA of V0s"};
  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f, 1.6f, 1.8f, 2.0f, 2.4f, 2.8f, 3.2f, 3.6f, 4.0f, 4.8f, 5.6f, 6.5f, 7.5f, 9.0f, 11.0f, 13.0f, 15.0f, 19.0f, 23.0f, 30.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisQuarkoniumMass{"axisQuarkoniumMass", {200, 3000.f, 3.200f}, "M (#Lambda, #bar{#Lambda} ) (GeV/#it{c}^{2})"};
  ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f}, "Centrality"};
  ConfigurableAxis axisNch{"axisNch", {500, 0.0f, +5000.0f}, "Number of charged particles"};

  ConfigurableAxis axisRawCentrality{"axisRawCentrality", {VARIABLE_WIDTH, 0.000f, 52.320f, 75.400f, 95.719f, 115.364f, 135.211f, 155.791f, 177.504f, 200.686f, 225.641f, 252.645f, 281.906f, 313.850f, 348.302f, 385.732f, 426.307f, 470.146f, 517.555f, 568.899f, 624.177f, 684.021f, 748.734f, 818.078f, 892.577f, 973.087f, 1058.789f, 1150.915f, 1249.319f, 1354.279f, 1465.979f, 1584.790f, 1710.778f, 1844.863f, 1985.746f, 2134.643f, 2291.610f, 2456.943f, 2630.653f, 2813.959f, 3006.631f, 3207.229f, 3417.641f, 3637.318f, 3865.785f, 4104.997f, 4354.938f, 4615.786f, 4885.335f, 5166.555f, 5458.021f, 5762.584f, 6077.881f, 6406.834f, 6746.435f, 7097.958f, 7462.579f, 7839.165f, 8231.629f, 8635.640f, 9052.000f, 9484.268f, 9929.111f, 10389.350f, 10862.059f, 11352.185f, 11856.823f, 12380.371f, 12920.401f, 13476.971f, 14053.087f, 14646.190f, 15258.426f, 15890.617f, 16544.433f, 17218.024f, 17913.465f, 18631.374f, 19374.983f, 20136.700f, 20927.783f, 21746.796f, 22590.880f, 23465.734f, 24372.274f, 25314.351f, 26290.488f, 27300.899f, 28347.512f, 29436.133f, 30567.840f, 31746.818f, 32982.664f, 34276.329f, 35624.859f, 37042.588f, 38546.609f, 40139.742f, 41837.980f, 43679.429f, 45892.130f, 400000.000f}, "raw centrality signal"}; // for QA

  ConfigurableAxis axisOccupancy{"axisOccupancy", {VARIABLE_WIDTH, 0.0f, 250.0f, 500.0f, 750.0f, 1000.0f, 1500.0f, 2000.0f, 3000.0f, 4500.0f, 6000.0f, 8000.0f, 10000.0f, 50000.0f}, "Occupancy"};

  // topological variable QA axes
  ConfigurableAxis axisDCAtoPV{"axisDCAtoPV", {20, 0.0f, 1.0f}, "DCA (cm)"};
  ConfigurableAxis axisDCAdau{"axisDCAdau", {20, 0.0f, 2.0f}, "DCA (cm)"};
  ConfigurableAxis axisPointingAngle{"axisPointingAngle", {20, 0.0f, 2.0f}, "pointing angle (rad)"};
  ConfigurableAxis axisV0Radius{"axisV0Radius", {20, 0.0f, 60.0f}, "V0 2D radius (cm)"};
  ConfigurableAxis axisV0MassWindow{"axisV0MassWindow", {40, -0.020f, 0.020f}, "V0 mass (GeV/#it{c}^{2})"};
  ConfigurableAxis axisNsigmaTPC{"axisNsigmaTPC", {200, -10.0f, 10.0f}, "N sigma TPC"};
  ConfigurableAxis axisTPCsignal{"axisTPCsignal", {200, 0.0f, 200.0f}, "TPC signal"};
  ConfigurableAxis axisTOFdeltaT{"axisTOFdeltaT", {200, -5000.0f, 5000.0f}, "TOF Delta T (ps)"};

  // UPC axes
  ConfigurableAxis axisSelGap{"axisSelGap", {4, -1.5, 2.5}, "Gap side"};

  // UPC selections
  SGSelector sgSelector;
  struct : ConfigurableGroup {
    Configurable<float> FV0cut{"upcCuts.FV0cut", 100., "FV0A threshold"};
    Configurable<float> FT0Acut{"upcCuts.FT0Acut", 200., "FT0A threshold"};
    Configurable<float> FT0Ccut{"upcCuts.FT0Ccut", 100., "FT0C threshold"};
    Configurable<float> ZDCcut{"upcCuts.ZDCcut", 10., "ZDC threshold"};
    // Configurable<float> gapSel{"upcCuts.gapSel", 2, "Gap selection"};
  } upcCuts;

  // AP plot axes
  ConfigurableAxis axisAPAlpha{"axisAPAlpha", {220, -1.1f, 1.1f}, "V0 AP alpha"};
  ConfigurableAxis axisAPQt{"axisAPQt", {220, 0.0f, 0.5f}, "V0 AP alpha"};

  // Track quality axes
  ConfigurableAxis axisTPCrows{"axisTPCrows", {160, 0.0f, 160.0f}, "N TPC rows"};
  ConfigurableAxis axisITSclus{"axisITSclus", {7, 0.0f, 7.0f}, "N ITS Clusters"};
  ConfigurableAxis axisITScluMap{"axisITSMap", {128, -0.5f, 127.5f}, "ITS Cluster map"};
  ConfigurableAxis axisDetMap{"axisDetMap", {16, -0.5f, 15.5f}, "Detector use map"};
  ConfigurableAxis axisITScluMapCoarse{"axisITScluMapCoarse", {16, -3.5f, 12.5f}, "ITS Coarse cluster map"};
  ConfigurableAxis axisDetMapCoarse{"axisDetMapCoarse", {5, -0.5f, 4.5f}, "Detector Coarse user map"};

  // MC coll assoc QA axis
  ConfigurableAxis axisMonteCarloNch{"axisMonteCarloNch", {300, 0.0f, 3000.0f}, "N_{ch} MC"};

  // PDG database
  Service<o2::framework::O2DatabasePDG> pdgDB;

  // For manual sliceBy
  PresliceUnsorted<soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraCollLabels>> perMcCollision = aod::v0data::straMCCollisionId;

  enum selection : uint64_t { selCosPA = 0,
                              selRadius,
                              selRadiusMax,
                              selDCANegToPV,
                              selDCAPosToPV,
                              selDCAV0ToPV,
                              selDCAV0Dau,
                              selK0ShortRapidity,
                              selLambdaRapidity,
                              selK0ShortMassWindow,
                              selLambdaMassWindow,
                              selAntiLambdaMassWindow,
                              selTPCPIDPositivePion,
                              selTPCPIDNegativePion,
                              selTPCPIDPositiveProton,
                              selTPCPIDNegativeProton,
                              selTOFDeltaTPositiveProtonLambda,
                              selTOFDeltaTPositivePionLambda,
                              selTOFDeltaTPositivePionK0Short,
                              selTOFDeltaTNegativeProtonLambda,
                              selTOFDeltaTNegativePionLambda,
                              selTOFDeltaTNegativePionK0Short,
                              selTOFNSigmaPositiveProtonLambda, // Nsigma
                              selTOFNSigmaPositivePionLambda,   // Nsigma
                              selTOFNSigmaPositivePionK0Short,  // Nsigma
                              selTOFNSigmaNegativeProtonLambda, // Nsigma
                              selTOFNSigmaNegativePionLambda,   // Nsigma
                              selTOFNSigmaNegativePionK0Short,  // Nsigma
                              selK0ShortCTau,
                              selLambdaCTau,
                              selK0ShortArmenteros,
                              selPosGoodTPCTrack, // at least min # TPC rows
                              selNegGoodTPCTrack, // at least min # TPC rows
                              selPosGoodITSTrack, // at least min # ITS clusters
                              selNegGoodITSTrack, // at least min # ITS clusters
                              selPosItsOnly,
                              selNegItsOnly,
                              selPosNotTPCOnly,
                              selNegNotTPCOnly,
                              selConsiderK0Short,    // for mc tagging
                              selConsiderLambda,     // for mc tagging
                              selConsiderAntiLambda, // for mc tagging
                              selPhysPrimK0Short,    // for mc tagging
                              selPhysPrimLambda,     // for mc tagging
                              selPhysPrimAntiLambda, // for mc tagging
  };

  uint64_t maskTopological;
  uint64_t maskTopoNoV0Radius;
  uint64_t maskTopoNoDCANegToPV;
  uint64_t maskTopoNoDCAPosToPV;
  uint64_t maskTopoNoCosPA;
  uint64_t maskTopoNoDCAV0Dau;
  uint64_t maskTopoNoDCAV0ToPV;
  uint64_t maskTrackProperties;

  uint64_t maskK0ShortSpecific;
  uint64_t maskLambdaSpecific;
  uint64_t maskAntiLambdaSpecific;

  uint64_t maskSelectionK0Short;
  uint64_t maskSelectionLambda;
  uint64_t maskSelectionAntiLambda;

  uint64_t secondaryMaskSelectionLambda;
  uint64_t secondaryMaskSelectionAntiLambda;

  void init(InitContext const&)
  {
    // initialise bit masks
    maskTopological = (uint64_t(1) << selCosPA) | (uint64_t(1) << selRadius) | (uint64_t(1) << selDCANegToPV) | (uint64_t(1) << selDCAPosToPV) | (uint64_t(1) << selDCAV0ToPV) | (uint64_t(1) << selDCAV0Dau) | (uint64_t(1) << selRadiusMax);
    maskTopoNoV0Radius = (uint64_t(1) << selCosPA) | (uint64_t(1) << selDCANegToPV) | (uint64_t(1) << selDCAPosToPV) | (uint64_t(1) << selDCAV0ToPV) | (uint64_t(1) << selDCAV0Dau) | (uint64_t(1) << selRadiusMax);
    maskTopoNoDCANegToPV = (uint64_t(1) << selCosPA) | (uint64_t(1) << selRadius) | (uint64_t(1) << selDCAPosToPV) | (uint64_t(1) << selDCAV0ToPV) | (uint64_t(1) << selDCAV0Dau) | (uint64_t(1) << selRadiusMax);
    maskTopoNoDCAPosToPV = (uint64_t(1) << selCosPA) | (uint64_t(1) << selRadius) | (uint64_t(1) << selDCANegToPV) | (uint64_t(1) << selDCAV0ToPV) | (uint64_t(1) << selDCAV0Dau) | (uint64_t(1) << selRadiusMax);
    maskTopoNoCosPA = (uint64_t(1) << selRadius) | (uint64_t(1) << selDCANegToPV) | (uint64_t(1) << selDCAPosToPV) | (uint64_t(1) << selDCAV0ToPV) | (uint64_t(1) << selDCAV0Dau) | (uint64_t(1) << selRadiusMax);
    maskTopoNoDCAV0Dau = (uint64_t(1) << selCosPA) | (uint64_t(1) << selRadius) | (uint64_t(1) << selDCANegToPV) | (uint64_t(1) << selDCAPosToPV) | (uint64_t(1) << selDCAV0ToPV) | (uint64_t(1) << selRadiusMax);
    maskTopoNoDCAV0ToPV = (uint64_t(1) << selCosPA) | (uint64_t(1) << selRadius) | (uint64_t(1) << selDCANegToPV) | (uint64_t(1) << selDCAPosToPV) | (uint64_t(1) << selDCAV0Dau) | (uint64_t(1) << selRadiusMax);

    maskK0ShortSpecific = (uint64_t(1) << selK0ShortRapidity) | (uint64_t(1) << selK0ShortCTau) | (uint64_t(1) << selK0ShortArmenteros) | (uint64_t(1) << selConsiderK0Short) | (uint64_t(1) << selK0ShortMassWindow);
    maskLambdaSpecific = (uint64_t(1) << selLambdaRapidity) | (uint64_t(1) << selLambdaCTau) | (uint64_t(1) << selConsiderLambda) | (uint64_t(1) << selLambdaMassWindow);
    maskAntiLambdaSpecific = (uint64_t(1) << selLambdaRapidity) | (uint64_t(1) << selLambdaCTau) | (uint64_t(1) << selConsiderAntiLambda) | (uint64_t(1) << selAntiLambdaMassWindow);

    // ask for specific TPC/TOF PID selections
    maskTrackProperties = 0;
    if (v0Selections.requirePosITSonly) {
      maskTrackProperties = maskTrackProperties | (uint64_t(1) << selPosItsOnly) | (uint64_t(1) << selPosGoodITSTrack);
    } else {
      maskTrackProperties = maskTrackProperties | (uint64_t(1) << selPosGoodTPCTrack) | (uint64_t(1) << selPosGoodITSTrack);
      // TPC signal is available: ask for positive track PID
      if (v0Selections.TpcPidNsigmaCut < 1e+5) { // safeguard for no cut
        maskK0ShortSpecific = maskK0ShortSpecific | (uint64_t(1) << selTPCPIDPositivePion);
        maskLambdaSpecific = maskLambdaSpecific | (uint64_t(1) << selTPCPIDPositiveProton);
        maskAntiLambdaSpecific = maskAntiLambdaSpecific | (uint64_t(1) << selTPCPIDPositivePion);
      }
      // TOF PID
      if (v0Selections.TofPidNsigmaCutK0Pi < 1e+5) // safeguard for no cut
        maskK0ShortSpecific = maskK0ShortSpecific | (uint64_t(1) << selTOFNSigmaPositivePionK0Short) | (uint64_t(1) << selTOFDeltaTPositivePionK0Short);
      if (v0Selections.TofPidNsigmaCutLaPr < 1e+5) // safeguard for no cut
        maskLambdaSpecific = maskLambdaSpecific | (uint64_t(1) << selTOFNSigmaPositiveProtonLambda) | (uint64_t(1) << selTOFDeltaTPositiveProtonLambda);
      if (v0Selections.TofPidNsigmaCutLaPi < 1e+5) // safeguard for no cut
        maskAntiLambdaSpecific = maskAntiLambdaSpecific | (uint64_t(1) << selTOFNSigmaPositivePionLambda) | (uint64_t(1) << selTOFDeltaTPositivePionLambda);
    }
    if (v0Selections.requireNegITSonly) {
      maskTrackProperties = maskTrackProperties | (uint64_t(1) << selNegItsOnly) | (uint64_t(1) << selNegGoodITSTrack);
    } else {
      maskTrackProperties = maskTrackProperties | (uint64_t(1) << selNegGoodTPCTrack) | (uint64_t(1) << selNegGoodITSTrack);
      // TPC signal is available: ask for negative track PID
      if (v0Selections.TpcPidNsigmaCut < 1e+5) { // safeguard for no cut
        maskK0ShortSpecific = maskK0ShortSpecific | (uint64_t(1) << selTPCPIDNegativePion);
        maskLambdaSpecific = maskLambdaSpecific | (uint64_t(1) << selTPCPIDNegativePion);
        maskAntiLambdaSpecific = maskAntiLambdaSpecific | (uint64_t(1) << selTPCPIDNegativeProton);
      }
      // TOF PID
      if (v0Selections.TofPidNsigmaCutK0Pi < 1e+5) // safeguard for no cut
        maskK0ShortSpecific = maskK0ShortSpecific | (uint64_t(1) << selTOFNSigmaNegativePionK0Short) | (uint64_t(1) << selTOFDeltaTNegativePionK0Short);
      if (v0Selections.TofPidNsigmaCutLaPi < 1e+5) // safeguard for no cut
        maskLambdaSpecific = maskLambdaSpecific | (uint64_t(1) << selTOFNSigmaNegativePionLambda) | (uint64_t(1) << selTOFDeltaTNegativePionLambda);
      if (v0Selections.TofPidNsigmaCutLaPr < 1e+5) // safeguard for no cut
        maskAntiLambdaSpecific = maskAntiLambdaSpecific | (uint64_t(1) << selTOFNSigmaNegativeProtonLambda) | (uint64_t(1) << selTOFDeltaTNegativeProtonLambda);
    }

    if (v0Selections.skipTPConly) {
      maskK0ShortSpecific = maskK0ShortSpecific | (uint64_t(1) << selPosNotTPCOnly) | (uint64_t(1) << selNegNotTPCOnly);
      maskLambdaSpecific = maskLambdaSpecific | (uint64_t(1) << selPosNotTPCOnly) | (uint64_t(1) << selNegNotTPCOnly);
      maskAntiLambdaSpecific = maskAntiLambdaSpecific | (uint64_t(1) << selPosNotTPCOnly) | (uint64_t(1) << selNegNotTPCOnly);
    }

    // Primary particle selection, central to analysis
    maskSelectionK0Short = maskTopological | maskTrackProperties | maskK0ShortSpecific | (uint64_t(1) << selPhysPrimK0Short);
    maskSelectionLambda = maskTopological | maskTrackProperties | maskLambdaSpecific | (uint64_t(1) << selPhysPrimLambda);
    maskSelectionAntiLambda = maskTopological | maskTrackProperties | maskAntiLambdaSpecific | (uint64_t(1) << selPhysPrimAntiLambda);

    // No primary requirement for feeddown matrix
    secondaryMaskSelectionLambda = maskTopological | maskTrackProperties | maskLambdaSpecific;
    secondaryMaskSelectionAntiLambda = maskTopological | maskTrackProperties | maskAntiLambdaSpecific;

    // Event Counters
    histos.add("hEventSelection", "hEventSelection", kTH1F, {{20, -0.5f, +19.5f}});
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(1, "All collisions");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(2, "sel8 cut");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(3, "posZ cut");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(4, "kNoITSROFrameBorder");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(5, "kNoTimeFrameBorder");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(6, "kIsVertexITSTPC");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(7, "kIsGoodZvtxFT0vsPV");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(8, "kIsVertexTOFmatched");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(9, "kIsVertexTRDmatched");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(10, "kNoSameBunchPileup");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(11, "kNoCollInTimeRangeStd");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(12, "kNoCollInTimeRangeNarrow");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(13, "Below min occup.");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(14, "Above max occup.");

    histos.add("hEventCentrality", "hEventCentrality", kTH1F, {{100, 0.0f, +100.0f}});
    histos.add("hCentralityVsNch", "hCentralityVsNch", kTH2F, {axisCentrality, axisNch});

    histos.add("hEventOccupancy", "hEventOccupancy", kTH1F, {axisOccupancy});
    histos.add("hCentralityVsOccupancy", "hCentralityVsOccupancy", kTH2F, {axisCentrality, axisOccupancy});

    if (!isPP) {
      histos.add("hGapSide", "Gap side; Entries", kTH1F, {{5, -0.5, 4.5}});
      histos.add("hSelGapSide", "Selected gap side; Entries", kTH1F, {axisSelGap});
      histos.add("hEventCentralityVsSelGapSide", ";Centrality (%); Selected gap side", kTH2F, {{100, 0.0f, +100.0f}, axisSelGap});
    }

    // for QA and test purposes
    auto hRawCentrality = histos.add<TH1>("hRawCentrality", "hRawCentrality", kTH1F, {axisRawCentrality});

    for (int ii = 1; ii < 101; ii++) {
      float value = 100.5f - static_cast<float>(ii);
      hRawCentrality->SetBinContent(ii, value);
    }

    // histograms versus mass
    histos.add("h3dMassDiV0s", "h3dMassDiV0s", kTH3F, {axisCentrality, axisPt, axisQuarkoniumMass});
    if (!isPP) {
      // Non-UPC info
      histos.add("h3dMassDiV0sHadronic", "h3dMassDiV0sHadronic", kTH3F, {axisCentrality, axisPt, axisQuarkoniumMass});
      // UPC info
      histos.add("h3dMassDiV0sSGA", "h3dMassDiV0sSGA", kTH3F, {axisCentrality, axisPt, axisQuarkoniumMass});
      histos.add("h3dMassDiV0sSGC", "h3dMassDiV0sSGC", kTH3F, {axisCentrality, axisPt, axisQuarkoniumMass});
      histos.add("h3dMassDiV0sDG", "h3dMassDiV0sDG", kTH3F, {axisCentrality, axisPt, axisQuarkoniumMass});
    }
    
    histos.add("h2dNbrOfK0ShortVsCentrality", "h2dNbrOfK0ShortVsCentrality", kTH2F, {axisCentrality, {10, -0.5f, 9.5f}}); 
    histos.add("h2dNbrOfLambdaVsCentrality", "h2dNbrOfLambdaVsCentrality", kTH2F, {axisCentrality, {10, -0.5f, 9.5f}});
    histos.add("h2dNbrOfAntiLambdaVsCentrality", "h2dNbrOfAntiLambdaVsCentrality", kTH2F, {axisCentrality, {10, -0.5f, 9.5f}}); 

    // if (calculateFeeddownMatrix && doprocessMonteCarlo) {
    //   histos.add("h3dLambdaFeeddown", "h3dLambdaFeeddown", kTH3F, {axisCentrality, axisV0Pt, axisCascPt});
    //   histos.add("h3dAntiLambdaFeeddown", "h3dAntiLambdaFeeddown", kTH3F, {axisCentrality, axisV0Pt, axisCascPt});
    // }

    // if (doTPCQA) {
    //   histos.add("K0Short/h3dPosNsigmaTPC", "h3dPosNsigmaTPC", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("K0Short/h3dNegNsigmaTPC", "h3dNegNsigmaTPC", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("K0Short/h3dPosTPCsignal", "h3dPosTPCsignal", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("K0Short/h3dNegTPCsignal", "h3dNegTPCsignal", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("K0Short/h3dPosNsigmaTPCvsTrackPtot", "h3dPosNsigmaTPCvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("K0Short/h3dNegNsigmaTPCvsTrackPtot", "h3dNegNsigmaTPCvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("K0Short/h3dPosTPCsignalVsTrackPtot", "h3dPosTPCsignalVsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("K0Short/h3dNegTPCsignalVsTrackPtot", "h3dNegTPCsignalVsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("K0Short/h3dPosNsigmaTPCvsTrackPt", "h3dPosNsigmaTPCvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("K0Short/h3dNegNsigmaTPCvsTrackPt", "h3dNegNsigmaTPCvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("K0Short/h3dPosTPCsignalVsTrackPt", "h3dPosTPCsignalVsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("K0Short/h3dNegTPCsignalVsTrackPt", "h3dNegTPCsignalVsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});

    //   histos.add("Lambda/h3dPosNsigmaTPC", "h3dPosNsigmaTPC", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("Lambda/h3dNegNsigmaTPC", "h3dNegNsigmaTPC", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("Lambda/h3dPosTPCsignal", "h3dPosTPCsignal", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("Lambda/h3dNegTPCsignal", "h3dNegTPCsignal", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("Lambda/h3dPosNsigmaTPCvsTrackPtot", "h3dPosNsigmaTPCvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("Lambda/h3dNegNsigmaTPCvsTrackPtot", "h3dNegNsigmaTPCvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("Lambda/h3dPosTPCsignalVsTrackPtot", "h3dPosTPCsignalVsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("Lambda/h3dNegTPCsignalVsTrackPtot", "h3dNegTPCsignalVsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("Lambda/h3dPosNsigmaTPCvsTrackPt", "h3dPosNsigmaTPCvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("Lambda/h3dNegNsigmaTPCvsTrackPt", "h3dNegNsigmaTPCvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("Lambda/h3dPosTPCsignalVsTrackPt", "h3dPosTPCsignalVsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("Lambda/h3dNegTPCsignalVsTrackPt", "h3dNegTPCsignalVsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});

    //   histos.add("AntiLambda/h3dPosNsigmaTPC", "h3dPosNsigmaTPC", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("AntiLambda/h3dNegNsigmaTPC", "h3dNegNsigmaTPC", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("AntiLambda/h3dPosTPCsignal", "h3dPosTPCsignal", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("AntiLambda/h3dNegTPCsignal", "h3dNegTPCsignal", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("AntiLambda/h3dPosNsigmaTPCvsTrackPtot", "h3dPosNsigmaTPCvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("AntiLambda/h3dNegNsigmaTPCvsTrackPtot", "h3dNegNsigmaTPCvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("AntiLambda/h3dPosTPCsignalVsTrackPtot", "h3dPosTPCsignalVsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("AntiLambda/h3dNegTPCsignalVsTrackPtot", "h3dNegTPCsignalVsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("AntiLambda/h3dPosNsigmaTPCvsTrackPt", "h3dPosNsigmaTPCvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("AntiLambda/h3dNegNsigmaTPCvsTrackPt", "h3dNegNsigmaTPCvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisNsigmaTPC});
    //   histos.add("AntiLambda/h3dPosTPCsignalVsTrackPt", "h3dPosTPCsignalVsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    //   histos.add("AntiLambda/h3dNegTPCsignalVsTrackPt", "h3dNegTPCsignalVsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTPCsignal});
    // }
    // if (doTOFQA) {
    //   histos.add("K0Short/h3dPosTOFdeltaT", "h3dPosTOFdeltaT", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("K0Short/h3dNegTOFdeltaT", "h3dNegTOFdeltaT", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("K0Short/h3dPosTOFdeltaTvsTrackPtot", "h3dPosTOFdeltaTvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("K0Short/h3dNegTOFdeltaTvsTrackPtot", "h3dNegTOFdeltaTvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("K0Short/h3dPosTOFdeltaTvsTrackPt", "h3dPosTOFdeltaTvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("K0Short/h3dNegTOFdeltaTvsTrackPt", "h3dNegTOFdeltaTvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});

    //   histos.add("Lambda/h3dPosTOFdeltaT", "h3dPosTOFdeltaT", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("Lambda/h3dNegTOFdeltaT", "h3dNegTOFdeltaT", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("Lambda/h3dPosTOFdeltaTvsTrackPtot", "h3dPosTOFdeltaTvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("Lambda/h3dNegTOFdeltaTvsTrackPtot", "h3dNegTOFdeltaTvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("Lambda/h3dPosTOFdeltaTvsTrackPt", "h3dPosTOFdeltaTvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("Lambda/h3dNegTOFdeltaTvsTrackPt", "h3dNegTOFdeltaTvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});

    //   histos.add("AntiLambda/h3dPosTOFdeltaT", "h3dPosTOFdeltaT", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("AntiLambda/h3dNegTOFdeltaT", "h3dNegTOFdeltaT", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("AntiLambda/h3dPosTOFdeltaTvsTrackPtot", "h3dPosTOFdeltaTvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("AntiLambda/h3dNegTOFdeltaTvsTrackPtot", "h3dNegTOFdeltaTvsTrackPtot", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("AntiLambda/h3dPosTOFdeltaTvsTrackPt", "h3dPosTOFdeltaTvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    //   histos.add("AntiLambda/h3dNegTOFdeltaTvsTrackPt", "h3dNegTOFdeltaTvsTrackPt", kTH3F, {axisCentrality, axisV0PtCoarse, axisTOFdeltaT});
    // }
    // if (doCollisionAssociationQA) {
    //   histos.add("K0Short/h2dPtVsNch", "h2dPtVsNch", kTH2F, {axisMonteCarloNch, axisV0Pt});
    //   histos.add("K0Short/h2dPtVsNch_BadCollAssig", "h2dPtVsNch_BadCollAssig", kTH2F, {axisMonteCarloNch, axisV0Pt});

    //   histos.add("Lambda/h2dPtVsNch", "h2dPtVsNch", kTH2F, {axisMonteCarloNch, axisV0Pt});
    //   histos.add("Lambda/h2dPtVsNch_BadCollAssig", "h2dPtVsNch_BadCollAssig", kTH2F, {axisMonteCarloNch, axisV0Pt});

    //   histos.add("AntiLambda/h2dPtVsNch", "h2dPtVsNch", kTH2F, {axisMonteCarloNch, axisV0Pt});
    //   histos.add("AntiLambda/h2dPtVsNch_BadCollAssig", "h2dPtVsNch_BadCollAssig", kTH2F, {axisMonteCarloNch, axisV0Pt});

    //   histos.add("h2dPtVsNch", "h2dPtVsNch", kTH2F, {axisMonteCarloNch, axisPt});
    //   histos.add("h2dPtVsNch_BadCollAssig", "h2dPtVsNch_BadCollAssig", kTH2F, {axisMonteCarloNch, axisPt});
    // }
    // if (doDetectPropQA) {
    //   histos.add("K0Short/h6dDetectPropVsCentrality", "h6dDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMapCoarse, axisITScluMapCoarse, axisDetMapCoarse, axisITScluMapCoarse, axisV0PtCoarse});
    //   histos.add("K0Short/h4dPosDetectPropVsCentrality", "h4dPosDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMap, axisITScluMap, axisV0PtCoarse});
    //   histos.add("K0Short/h4dNegDetectPropVsCentrality", "h4dNegDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMap, axisITScluMap, axisV0PtCoarse});

    //   histos.add("Lambda/h6dDetectPropVsCentrality", "h6dDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMapCoarse, axisITScluMapCoarse, axisDetMapCoarse, axisITScluMapCoarse, axisV0PtCoarse});
    //   histos.add("Lambda/h4dPosDetectPropVsCentrality", "h4dPosDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMap, axisITScluMap, axisV0PtCoarse});
    //   histos.add("Lambda/h4dNegDetectPropVsCentrality", "h4dNegDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMap, axisITScluMap, axisV0PtCoarse});

    //   histos.add("AntiLambda/h6dDetectPropVsCentrality", "h6dDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMapCoarse, axisITScluMapCoarse, axisDetMapCoarse, axisITScluMapCoarse, axisV0PtCoarse});
    //   histos.add("AntiLambda/h4dPosDetectPropVsCentrality", "h4dPosDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMap, axisITScluMap, axisV0PtCoarse});
    //   histos.add("AntiLambda/h4dNegDetectPropVsCentrality", "h4dNegDetectPropVsCentrality", kTHnF, {axisCentrality, axisDetMap, axisITScluMap, axisV0PtCoarse});
    // }

    // // QA histograms if requested
    // if (doCompleteTopoQA) {
    //   // initialize for K0short...
    //   histos.add("K0Short/h3dPosDCAToPV", "h3dPosDCAToPV", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAtoPV});
    //   histos.add("K0Short/h3dNegDCAToPV", "h3dNegDCAToPV", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAtoPV});
    //   histos.add("K0Short/h3dDCADaughters", "h3dDCADaughters", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAdau});
    //   histos.add("K0Short/h3dPointingAngle", "h3dPointingAngle", kTHnF, {axisCentrality, axisV0PtCoarse, axisPointingAngle});
    //   histos.add("K0Short/h3dV0Radius", "h3dV0Radius", kTHnF, {axisCentrality, axisV0PtCoarse, axisV0Radius});

    //   histos.add("Lambda/h3dPosDCAToPV", "h3dPosDCAToPV", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAtoPV});
    //   histos.add("Lambda/h3dNegDCAToPV", "h3dNegDCAToPV", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAtoPV});
    //   histos.add("Lambda/h3dDCADaughters", "h3dDCADaughters", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAdau});
    //   histos.add("Lambda/h3dPointingAngle", "h3dPointingAngle", kTHnF, {axisCentrality, axisV0PtCoarse, axisPointingAngle});
    //   histos.add("Lambda/h3dV0Radius", "h3dV0Radius", kTHnF, {axisCentrality, axisV0PtCoarse, axisV0Radius});
      
    //   histos.add("AntiLambda/h3dPosDCAToPV", "h3dPosDCAToPV", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAtoPV});
    //   histos.add("AntiLambda/h3dNegDCAToPV", "h3dNegDCAToPV", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAtoPV});
    //   histos.add("AntiLambda/h3dDCADaughters", "h3dDCADaughters", kTHnF, {axisCentrality, axisV0PtCoarse, axisDCAdau});
    //   histos.add("AntiLambda/h3dPointingAngle", "h3dPointingAngle", kTHnF, {axisCentrality, axisV0PtCoarse, axisPointingAngle});
    //   histos.add("AntiLambda/h3dV0Radius", "h3dV0Radius", kTHnF, {axisCentrality, axisV0PtCoarse, axisV0Radius});
    // }

    // if (doPlainTopoQA) {
    //   // All candidates received
    //   histos.add("hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("hDCADaughters", "hDCADaughters", kTH1F, {axisDCAdau});
    //   histos.add("hPointingAngle", "hPointingAngle", kTH1F, {axisPointingAngle});
    //   histos.add("hV0Radius", "hV0Radius", kTH1F, {axisV0Radius});
    //   histos.add("hInvMassK0Short", "hInvMassK0Short", kTH1F, {axisV0MassWindow});
    //   histos.add("hInvMassLambda", "hInvMassLambda", kTH1F, {axisV0MassWindow});
    //   histos.add("hInvMassAntiLambda", "hInvMassAntiLambda", kTH1F, {axisV0MassWindow});
    //   histos.add("h2dArmenterosAll", "h2dArmenterosAll", kTH2F, {axisAPAlpha, axisAPQt});
    //   histos.add("h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    //   histos.add("h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});

    //   // Candidates after K0Short selections
    //   histos.add("K0Short/hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("K0Short/hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("K0Short/hDCADaughters", "hDCADaughters", kTH1F, {axisDCAdau});
    //   histos.add("K0Short/hPointingAngle", "hPointingAngle", kTH1F, {axisPointingAngle});
    //   histos.add("K0Short/hV0Radius", "hV0Radius", kTH1F, {axisV0Radius});
    //   histos.add("K0Short/hInvMassWindow", "hInvMassWindow", kTH1F, {axisV0MassWindow});
    //   histos.add("K0Short/h2dArmenterosSelected", "h2dArmenterosSelected", kTH2F, {axisAPAlpha, axisAPQt});
    //   histos.add("K0Short/h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    //   histos.add("K0Short/h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
      
    //   // Candidates after Lambda selections
    //   histos.add("Lambda/hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("Lambda/hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("Lambda/hDCADaughters", "hDCADaughters", kTH1F, {axisDCAdau});
    //   histos.add("Lambda/hPointingAngle", "hPointingAngle", kTH1F, {axisPointingAngle});
    //   histos.add("Lambda/hV0Radius", "hV0Radius", kTH1F, {axisV0Radius});
    //   histos.add("Lambda/hInvMassWindow", "hInvMassWindow", kTH1F, {axisV0MassWindow});
    //   histos.add("Lambda/h2dArmenterosSelected", "h2dArmenterosSelected", kTH2F, {axisAPAlpha, axisAPQt});
    //   histos.add("Lambda/h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    //   histos.add("Lambda/h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});

    //   // Candidates after AntiLambda selections
    //   histos.add("AntiLambda/hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("AntiLambda/hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    //   histos.add("AntiLambda/hDCADaughters", "hDCADaughters", kTH1F, {axisDCAdau});
    //   histos.add("AntiLambda/hPointingAngle", "hPointingAngle", kTH1F, {axisPointingAngle});
    //   histos.add("AntiLambda/hV0Radius", "hV0Radius", kTH1F, {axisV0Radius});
    //   histos.add("AntiLambda/hInvMassWindow", "hInvMassWindow", kTH1F, {axisV0MassWindow});
    //   histos.add("AntiLambda/h2dArmenterosSelected", "h2dArmenterosSelected", kTH2F, {axisAPAlpha, axisAPQt});
    //   histos.add("AntiLambda/h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    //   histos.add("AntiLambda/h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    // }

    // Creation of histograms: MC generated
    // if (doprocessGenerated) {
    //   histos.add("hGenEvents", "hGenEvents", kTH2F, {{axisNch}, {2, -0.5f, +1.5f}});
    //   histos.get<TH2>(HIST("hGenEvents"))->GetYaxis()->SetBinLabel(1, "All gen. events");
    //   histos.get<TH2>(HIST("hGenEvents"))->GetYaxis()->SetBinLabel(2, "Gen. with at least 1 rec. events");
    //   histos.add("hGenEventCentrality", "hGenEventCentrality", kTH1F, {{100, 0.0f, +100.0f}});

    //   histos.add("hCentralityVsNcoll_beforeEvSel", "hCentralityVsNcoll_beforeEvSel", kTH2F, {axisCentrality, {50, -0.5f, 49.5f}});
    //   histos.add("hCentralityVsNcoll_afterEvSel", "hCentralityVsNcoll_afterEvSel", kTH2F, {axisCentrality, {50, -0.5f, 49.5f}});

    //   histos.add("hCentralityVsMultMC", "hCentralityVsMultMC", kTH2F, {{100, 0.0f, 100.0f}, axisNch});

    //   histos.add("h2dGenK0Short", "h2dGenK0Short", kTH2D, {axisCentrality, axisV0Pt});
    //   histos.add("h2dGenLambda", "h2dGenLambda", kTH2D, {axisCentrality, axisV0Pt});
    //   histos.add("h2dGenAntiLambda", "h2dGenAntiLambda", kTH2D, {axisCentrality, axisV0Pt});
    //   histos.add("h2dGenXiMinus", "h2dGenXiMinus", kTH2D, {axisCentrality, axisV0Pt});
    //   histos.add("h2dGenXiPlus", "h2dGenXiPlus", kTH2D, {axisCentrality, axisV0Pt});
    //   histos.add("h2dGenOmegaMinus", "h2dGenOmegaMinus", kTH2D, {axisCentrality, axisV0Pt});
    //   histos.add("h2dGenOmegaPlus", "h2dGenOmegaPlus", kTH2D, {axisCentrality, axisV0Pt});

    //   histos.add("h2dGenK0ShortVsMultMC", "h2dGenK0ShortVsMultMC", kTH2D, {axisNch, axisV0Pt});
    //   histos.add("h2dGenLambdaVsMultMC", "h2dGenLambdaVsMultMC", kTH2D, {axisNch, axisV0Pt});
    //   histos.add("h2dGenAntiLambdaVsMultMC", "h2dGenAntiLambdaVsMultMC", kTH2D, {axisNch, axisV0Pt});
    //   histos.add("h2dGenXiMinusVsMultMC", "h2dGenXiMinusVsMultMC", kTH2D, {axisNch, axisV0Pt});
    //   histos.add("h2dGenXiPlusVsMultMC", "h2dGenXiPlusVsMultMC", kTH2D, {axisNch, axisV0Pt});
    //   histos.add("h2dGenOmegaMinusVsMultMC", "h2dGenOmegaMinusVsMultMC", kTH2D, {axisNch, axisV0Pt});
    //   histos.add("h2dGenOmegaPlusVsMultMC", "h2dGenOmegaPlusVsMultMC", kTH2D, {axisNch, axisV0Pt});
    // }

    // inspect histogram sizes, please
    histos.print();
  }

  void initCCDB(soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraStamps>::iterator const& collision)
  {
    if (mRunNumber == collision.runNumber()) {
      return;
    }

    mRunNumber = collision.runNumber();

    // machine learning initialization if requested
    if (mlConfigurations.calculateK0ShortScores ||
        mlConfigurations.calculateLambdaScores ||
        mlConfigurations.calculateAntiLambdaScores) {
      int64_t timeStampML = collision.timestamp();
      if (mlConfigurations.timestampCCDB.value != -1)
        timeStampML = mlConfigurations.timestampCCDB.value;
      LoadMachines(timeStampML);
    }
  }

  template <typename TV0, typename TCollision>
  uint64_t computeReconstructionBitmap(TV0 v0, TCollision collision, float rapidityLambda, float rapidityK0Short, float /*pT*/)
  // precalculate this information so that a check is one mask operation, not many
  {
    uint64_t bitMap = 0;

    //
    // Base topological variables
    //

    // v0 radius min/max selections
    if (v0.v0radius() > v0Selections.v0radius)
      bitset(bitMap, selRadius);
    if (v0.v0radius() < v0Selections.v0radiusMax)
      bitset(bitMap, selRadiusMax);
    // DCA proton and pion to PV for Lambda and AntiLambda decay hypotheses
    if (TMath::Abs(v0.dcapostopv()) > v0Selections.dcaprotontopv &&
        TMath::Abs(v0.dcanegtopv()) > v0Selections.dcapiontopv) {
      bitset(bitMap, selDCAPosToPV);
      bitset(bitMap, selDCANegToPV);
    } else if (TMath::Abs(v0.dcapostopv()) > v0Selections.dcapiontopv &&
               TMath::Abs(v0.dcanegtopv()) > v0Selections.dcaprotontopv) {
      bitset(bitMap, selDCAPosToPV);
      bitset(bitMap, selDCANegToPV);
    }
    // V0 cosine of pointing angle
    if (v0.v0cosPA() > v0Selections.v0cospa)
      bitset(bitMap, selCosPA);
    // DCA between v0 daughters
    if (v0.dcaV0daughters() < v0Selections.dcav0dau)
      bitset(bitMap, selDCAV0Dau);
    // DCA V0 to prim vtx
    if (v0.dcav0topv() > v0Selections.dcav0topv)
      bitset(bitMap, selDCAV0ToPV);

    //
    // rapidity
    //
    if (TMath::Abs(rapidityLambda) < v0Selections.rapidityCut)
      bitset(bitMap, selLambdaRapidity);
    if (TMath::Abs(rapidityK0Short) < v0Selections.rapidityCut)
      bitset(bitMap, selK0ShortRapidity);

    //
    // invariant mass window
    //
    if (TMath::Abs(v0.mK0Short() - pdgDB->Mass(310)) < v0Selections.v0MassWindow)
      bitset(bitMap, selK0ShortMassWindow);
    if (TMath::Abs(v0.mLambda() - pdgDB->Mass(3122)) < v0Selections.v0MassWindow)
      bitset(bitMap, selLambdaMassWindow);
    if (TMath::Abs(v0.mAntiLambda() - pdgDB->Mass(3122)) < v0Selections.v0MassWindow)
      bitset(bitMap, selAntiLambdaMassWindow);

    auto posTrackExtra = v0.template posTrackExtra_as<dauTracks>();
    auto negTrackExtra = v0.template negTrackExtra_as<dauTracks>();

    //
    // ITS quality flags
    //
    if (posTrackExtra.itsNCls() >= v0Selections.minITSclusters)
      bitset(bitMap, selPosGoodITSTrack);
    if (negTrackExtra.itsNCls() >= v0Selections.minITSclusters)
      bitset(bitMap, selNegGoodITSTrack);

    //
    // TPC quality flags
    //
    if (posTrackExtra.tpcCrossedRows() >= v0Selections.minTPCrows)
      bitset(bitMap, selPosGoodTPCTrack);
    if (negTrackExtra.tpcCrossedRows() >= v0Selections.minTPCrows)
      bitset(bitMap, selNegGoodTPCTrack);

    //
    // TPC PID
    //
    if (fabs(posTrackExtra.tpcNSigmaPi()) < v0Selections.TpcPidNsigmaCut)
      bitset(bitMap, selTPCPIDPositivePion);
    if (fabs(posTrackExtra.tpcNSigmaPr()) < v0Selections.TpcPidNsigmaCut)
      bitset(bitMap, selTPCPIDPositiveProton);
    if (fabs(negTrackExtra.tpcNSigmaPi()) < v0Selections.TpcPidNsigmaCut)
      bitset(bitMap, selTPCPIDNegativePion);
    if (fabs(negTrackExtra.tpcNSigmaPr()) < v0Selections.TpcPidNsigmaCut)
      bitset(bitMap, selTPCPIDNegativeProton);

    //
    // TOF PID in DeltaT
    // Positive track
    if (fabs(v0.posTOFDeltaTLaPr()) < v0Selections.maxDeltaTimeProton)
      bitset(bitMap, selTOFDeltaTPositiveProtonLambda);
    if (fabs(v0.posTOFDeltaTLaPi()) < v0Selections.maxDeltaTimePion)
      bitset(bitMap, selTOFDeltaTPositivePionLambda);
    if (fabs(v0.posTOFDeltaTK0Pi()) < v0Selections.maxDeltaTimePion)
      bitset(bitMap, selTOFDeltaTPositivePionK0Short);
    // Negative track
    if (fabs(v0.negTOFDeltaTLaPr()) < v0Selections.maxDeltaTimeProton)
      bitset(bitMap, selTOFDeltaTNegativeProtonLambda);
    if (fabs(v0.negTOFDeltaTLaPi()) < v0Selections.maxDeltaTimePion)
      bitset(bitMap, selTOFDeltaTNegativePionLambda);
    if (fabs(v0.negTOFDeltaTK0Pi()) < v0Selections.maxDeltaTimePion)
      bitset(bitMap, selTOFDeltaTNegativePionK0Short);

    //
    // TOF PID in NSigma
    // Positive track
    if (fabs(v0.tofNSigmaLaPr()) < v0Selections.TofPidNsigmaCutLaPr)
      bitset(bitMap, selTOFNSigmaPositiveProtonLambda);
    if (fabs(v0.tofNSigmaALaPi()) < v0Selections.TofPidNsigmaCutLaPi)
      bitset(bitMap, selTOFNSigmaPositivePionLambda);
    if (fabs(v0.tofNSigmaK0PiPlus()) < v0Selections.TofPidNsigmaCutK0Pi)
      bitset(bitMap, selTOFNSigmaPositivePionK0Short);
    // Negative track
    if (fabs(v0.tofNSigmaALaPr()) < v0Selections.TofPidNsigmaCutLaPr)
      bitset(bitMap, selTOFNSigmaNegativeProtonLambda);
    if (fabs(v0.tofNSigmaLaPi()) < v0Selections.TofPidNsigmaCutLaPi)
      bitset(bitMap, selTOFNSigmaNegativePionLambda);
    if (fabs(v0.tofNSigmaK0PiMinus()) < v0Selections.TofPidNsigmaCutK0Pi)
      bitset(bitMap, selTOFNSigmaNegativePionK0Short);

    //
    // ITS only tag
    if (posTrackExtra.tpcCrossedRows() < 1)
      bitset(bitMap, selPosItsOnly);
    if (negTrackExtra.tpcCrossedRows() < 1)
      bitset(bitMap, selNegItsOnly);

    //
    // TPC only tag
    if (posTrackExtra.detectorMap() != o2::aod::track::TPC)
      bitset(bitMap, selPosNotTPCOnly);
    if (negTrackExtra.detectorMap() != o2::aod::track::TPC)
      bitset(bitMap, selNegNotTPCOnly);

    //
    // proper lifetime
    if (v0.distovertotmom(collision.posX(), collision.posY(), collision.posZ()) * o2::constants::physics::MassLambda0 < lifetimecut->get("lifetimecutLambda"))
      bitset(bitMap, selLambdaCTau);
    if (v0.distovertotmom(collision.posX(), collision.posY(), collision.posZ()) * o2::constants::physics::MassK0Short < lifetimecut->get("lifetimecutK0S"))
      bitset(bitMap, selK0ShortCTau);

    //
    // armenteros
    if (v0.qtarm() * v0Selections.armPodCut > TMath::Abs(v0.alpha()) || v0Selections.armPodCut < 1e-4)
      bitset(bitMap, selK0ShortArmenteros);

    return bitMap;
  }

  template <typename TV0>
  uint64_t computeMCAssociation(TV0 v0)
  // precalculate this information so that a check is one mask operation, not many
  {
    uint64_t bitMap = 0;
    // check for specific particle species

    if (v0.pdgCode() == 310 && v0.pdgCodePositive() == 211 && v0.pdgCodeNegative() == -211) {
      bitset(bitMap, selConsiderK0Short);
      if (v0.isPhysicalPrimary())
        bitset(bitMap, selPhysPrimK0Short);
    }
    if (v0.pdgCode() == 3122 && v0.pdgCodePositive() == 2212 && v0.pdgCodeNegative() == -211) {
      bitset(bitMap, selConsiderLambda);
      if (v0.isPhysicalPrimary())
        bitset(bitMap, selPhysPrimLambda);
    }
    if (v0.pdgCode() == -3122 && v0.pdgCodePositive() == 211 && v0.pdgCodeNegative() == -2212) {
      bitset(bitMap, selConsiderAntiLambda);
      if (v0.isPhysicalPrimary())
        bitset(bitMap, selPhysPrimAntiLambda);
    }
    return bitMap;
  }

  bool verifyMask(uint64_t bitmap, uint64_t mask)
  {
    return (bitmap & mask) == mask;
  }

  // function to load models for ML-based classifiers
  void LoadMachines(int64_t timeStampML)
  {
    if (mlConfigurations.loadCustomModelsFromCCDB) {
      ccdbApi.init(ccdbConfigurations.ccdburl);
      LOG(info) << "Fetching models for timestamp: " << timeStampML;

      if (mlConfigurations.calculateLambdaScores) {
        bool retrieveSuccessLambda = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathLambda.value);
        if (retrieveSuccessLambda) {
          mlCustomModelLambda.initModel(mlConfigurations.localModelPathLambda.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the Lambda model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }

      if (mlConfigurations.calculateAntiLambdaScores) {
        bool retrieveSuccessAntiLambda = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathAntiLambda.value);
        if (retrieveSuccessAntiLambda) {
          mlCustomModelAntiLambda.initModel(mlConfigurations.localModelPathAntiLambda.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the AntiLambda model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }

      if (mlConfigurations.calculateK0ShortScores) {
        bool retrieveSuccessKZeroShort = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathK0Short.value);
        if (retrieveSuccessKZeroShort) {
          mlCustomModelK0Short.initModel(mlConfigurations.localModelPathK0Short.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the K0Short model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }
    } else {
      if (mlConfigurations.calculateLambdaScores)
        mlCustomModelLambda.initModel(mlConfigurations.localModelPathLambda.value, mlConfigurations.enableOptimizations.value);
      if (mlConfigurations.calculateAntiLambdaScores)
        mlCustomModelAntiLambda.initModel(mlConfigurations.localModelPathAntiLambda.value, mlConfigurations.enableOptimizations.value);
      if (mlConfigurations.calculateK0ShortScores)
        mlCustomModelK0Short.initModel(mlConfigurations.localModelPathK0Short.value, mlConfigurations.enableOptimizations.value);
    }
    LOG(info) << "ML Models loaded.";
  }

  template <typename TV0>
  void analyseV0Candidate(TV0 v0, float pt, float centrality, uint64_t selMap, std::vector<bool>& selK0ShortIndices, std::vector<bool>& selLambdaIndices, std::vector<bool>& selAntiLambdaIndices, int v0TableOffset)
  // precalculate this information so that a check is one mask operation, not many
  {
    bool passK0ShortSelections = false;
    bool passLambdaSelections = false;
    bool passAntiLambdaSelections = false;

    // machine learning is on, go for calculation of thresholds
    // FIXME THIS NEEDS ADJUSTING
    std::vector<float> inputFeatures{pt, 0.0f, 0.0f, v0.v0radius(), v0.v0cosPA(), v0.dcaV0daughters(), v0.dcapostopv(), v0.dcanegtopv()};

    if (mlConfigurations.useK0ShortScores) {
      float k0shortScore = -1;
      if (mlConfigurations.calculateK0ShortScores) {
        // evaluate machine-learning scores
        float* k0shortProbability = mlCustomModelK0Short.evalModel(inputFeatures);
        k0shortScore = k0shortProbability[1];
      } else {
        k0shortScore = v0.k0ShortBDTScore();
      }
      if (k0shortScore > mlConfigurations.thresholdK0Short.value) {
        passK0ShortSelections = true;
      }
    } else {
      passK0ShortSelections = verifyMask(selMap, maskSelectionK0Short);
    }
    if (mlConfigurations.useLambdaScores) {
      float lambdaScore = -1;
      if (mlConfigurations.calculateLambdaScores) {
        // evaluate machine-learning scores
        float* lambdaProbability = mlCustomModelLambda.evalModel(inputFeatures);
        lambdaScore = lambdaProbability[1];
      } else {
        lambdaScore = v0.lambdaBDTScore();
      }
      if (lambdaScore > mlConfigurations.thresholdK0Short.value) {
        passLambdaSelections = true;
      }
    } else {
      passLambdaSelections = verifyMask(selMap, maskSelectionLambda);
    }
    if (mlConfigurations.useLambdaScores) {
      float antiLambdaScore = -1;
      if (mlConfigurations.calculateAntiLambdaScores) {
        // evaluate machine-learning scores
        float* antilambdaProbability = mlCustomModelAntiLambda.evalModel(inputFeatures);
        antiLambdaScore = antilambdaProbability[1];
      } else {
        antiLambdaScore = v0.antiLambdaBDTScore();
      }
      if (antiLambdaScore > mlConfigurations.thresholdK0Short.value) {
        passAntiLambdaSelections = true;
      }
    } else {
      passAntiLambdaSelections = verifyMask(selMap, maskSelectionAntiLambda);
    }

    // need local index because of the grouping of collisions
    selK0ShortIndices[v0.globalIndex() - v0TableOffset] = passK0ShortSelections; 
    selLambdaIndices[v0.globalIndex() - v0TableOffset] = passLambdaSelections;
    selAntiLambdaIndices[v0.globalIndex() - v0TableOffset] = passAntiLambdaSelections;

    // auto posTrackExtra = v0.template posTrackExtra_as<dauTracks>();
    // auto negTrackExtra = v0.template negTrackExtra_as<dauTracks>();

    // // __________________________________________
    // // fill with no selection if plain QA requested
    // if (doPlainTopoQA) {
    //   histos.fill(HIST("hPosDCAToPV"), v0.dcapostopv());
    //   histos.fill(HIST("hNegDCAToPV"), v0.dcanegtopv());
    //   histos.fill(HIST("hDCADaughters"), v0.dcaV0daughters());
    //   histos.fill(HIST("hPointingAngle"), TMath::ACos(v0.v0cosPA()));
    //   histos.fill(HIST("hV0Radius"), v0.v0radius());
    //   histos.fill(HIST("hInvMassK0Short"), v0.mK0Short() - pdgDB->Mass(310));
    //   histos.fill(HIST("hInvMassLambda"), v0.mLambda() - pdgDB->Mass(3122));
    //   histos.fill(HIST("hInvMassAntiLambda"), v0.mAntiLambda() - pdgDB->Mass(3122));
    //   histos.fill(HIST("h2dArmenterosAll"), v0.alpha(), v0.qtarm());
    //   histos.fill(HIST("h2dPositiveITSvsTPCpts"), posTrackExtra.tpcCrossedRows(), posTrackExtra.itsNCls());
    //   histos.fill(HIST("h2dNegativeITSvsTPCpts"), negTrackExtra.tpcCrossedRows(), negTrackExtra.itsNCls());
    // }

    // // __________________________________________
    // // do systematics / qa plots
    // if (doCompleteTopoQA) {
    //   // K0Short case
    //   if (verifyMask(selMap, maskTopoNoV0Radius | maskK0ShortSpecific))
    //     histos.fill(HIST("K0Short/h3dV0Radius"), centrality, pt, v0.v0radius());
    //   if (verifyMask(selMap, maskTopoNoDCAPosToPV | maskK0ShortSpecific))
    //     histos.fill(HIST("K0Short/h3dPosDCAToPV"), centrality, pt, TMath::Abs(v0.dcapostopv()));
    //   if (verifyMask(selMap, maskTopoNoDCANegToPV | maskK0ShortSpecific))
    //     histos.fill(HIST("K0Short/h3dNegDCAToPV"), centrality, pt, TMath::Abs(v0.dcanegtopv()));
    //   if (verifyMask(selMap, maskTopoNoCosPA | maskK0ShortSpecific))
    //     histos.fill(HIST("K0Short/h3dPointingAngle"), centrality, pt, TMath::ACos(v0.v0cosPA()));
    //   if (verifyMask(selMap, maskTopoNoDCAV0Dau | maskK0ShortSpecific))
    //     histos.fill(HIST("K0Short/h3dDCADaughters"), centrality, pt, v0.dcaV0daughters());

    //   // Lambda case
    //   if (verifyMask(selMap, maskTopoNoV0Radius | maskLambdaSpecific))
    //     histos.fill(HIST("Lambda/h3dV0Radius"), centrality, pt, v0.v0radius());
    //   if (verifyMask(selMap, maskTopoNoDCAPosToPV | maskLambdaSpecific))
    //     histos.fill(HIST("Lambda/h3dPosDCAToPV"), centrality, pt, TMath::Abs(v0.dcapostopv()));
    //   if (verifyMask(selMap, maskTopoNoDCANegToPV | maskLambdaSpecific))
    //     histos.fill(HIST("Lambda/h3dNegDCAToPV"), centrality, pt, TMath::Abs(v0.dcanegtopv()));
    //   if (verifyMask(selMap, maskTopoNoCosPA | maskLambdaSpecific))
    //     histos.fill(HIST("Lambda/h3dPointingAngle"), centrality, pt, TMath::ACos(v0.v0cosPA()));
    //   if (verifyMask(selMap, maskTopoNoDCAV0Dau | maskLambdaSpecific))
    //     histos.fill(HIST("Lambda/h3dDCADaughters"), centrality, pt, v0.dcaV0daughters());
      
    //   // AntiLambda case
    //   if (verifyMask(selMap, maskTopoNoV0Radius | maskAntiLambdaSpecific))
    //     histos.fill(HIST("AntiLambda/h3dV0Radius"), centrality, pt, v0.v0radius());
    //   if (verifyMask(selMap, maskTopoNoDCAPosToPV | maskAntiLambdaSpecific))
    //     histos.fill(HIST("AntiLambda/h3dPosDCAToPV"), centrality, pt, TMath::Abs(v0.dcapostopv()));
    //   if (verifyMask(selMap, maskTopoNoDCANegToPV | maskAntiLambdaSpecific))
    //     histos.fill(HIST("AntiLambda/h3dNegDCAToPV"), centrality, pt, TMath::Abs(v0.dcanegtopv()));
    //   if (verifyMask(selMap, maskTopoNoCosPA | maskAntiLambdaSpecific))
    //     histos.fill(HIST("AntiLambda/h3dPointingAngle"), centrality, pt, TMath::ACos(v0.v0cosPA()));
    //   if (verifyMask(selMap, maskTopoNoDCAV0Dau | maskAntiLambdaSpecific))
    //     histos.fill(HIST("AntiLambda/h3dDCADaughters"), centrality, pt, v0.dcaV0daughters());
    // } // end systematics / qa
  }

  // template <typename TV0>
  // void fillV0sInfo(TV0 lambda, TV0 antiLambda, float centrality)
  // // fill information about V0 daughter
  // {
  //   // Lambda part
  //   auto lambdaPosTrackExtra = lambda.template posTrackExtra_as<dauTracks>();
  //   auto lambdaNegTrackExtra = lambda.template negTrackExtra_as<dauTracks>();

  //   bool lambdaPosIsFromAfterburner = lambdaPosTrackExtra.itsChi2PerNcl() < 0;
  //   bool lambdaNegIsFromAfterburner = lambdaNegTrackExtra.itsChi2PerNcl() < 0;

  //   uint lambdaPosDetMap = computeDetBitmap(lambdaPosTrackExtra.detectorMap());
  //   int lambdaPosITSclusMap = computeITSclusBitmap(lambdaPosTrackExtra.itsClusterMap(), lambdaPosIsFromAfterburner);
  //   uint lambdaNegDetMap = computeDetBitmap(lambdaNegTrackExtra.detectorMap());
  //   int lambdaNegITSclusMap = computeITSclusBitmap(lambdaNegTrackExtra.itsClusterMap(), lambdaNegIsFromAfterburner);
  //   // __________________________________________
  //   // main analysis
  //   if (doPlainTopoQA) {
  //     histos.fill(HIST("Lambda/hInvMassWindow"), lambda.mLambda()-pdgDB->Mass(3122));
  //     histos.fill(HIST("Lambda/hPosDCAToPV"), lambda.dcapostopv());
  //     histos.fill(HIST("Lambda/hNegDCAToPV"), lambda.dcanegtopv());
  //     histos.fill(HIST("Lambda/hDCADaughters"), lambda.dcaV0daughters());
  //     histos.fill(HIST("Lambda/hPointingAngle"), TMath::ACos(lambda.v0cosPA()));
  //     histos.fill(HIST("Lambda/hV0Radius"), lambda.v0radius());
  //     histos.fill(HIST("Lambda/h2dArmenterosSelected"), lambda.alpha(), lambda.qtarm()); // cross-check
  //     histos.fill(HIST("Lambda/h2dPositiveITSvsTPCpts"), lambdaPosTrackExtra.tpcCrossedRows(), lambdaPosTrackExtra.itsNCls());
  //     histos.fill(HIST("Lambda/h2dNegativeITSvsTPCpts"), lambdaNegTrackExtra.tpcCrossedRows(), lambdaNegTrackExtra.itsNCls());
  //   }
  //   if (doDetectPropQA) {
  //     histos.fill(HIST("Lambda/h6dDetectPropVsCentrality"), centrality, lambdaPosDetMap, lambdaPosITSclusMap, lambdaNegDetMap, lambdaNegITSclusMap, lambda.pt());
  //     histos.fill(HIST("Lambda/h4dPosDetectPropVsCentrality"), centrality, lambdaPosTrackExtra.detectorMap(), lambdaPosTrackExtra.itsClusterMap(), lambda.pt());
  //     histos.fill(HIST("Lambda/h4dNegDetectPropVsCentrality"), centrality, lambdaNegTrackExtra.detectorMap(), lambdaNegTrackExtra.itsClusterMap(), lambda.pt());
  //   }
  //   if (doTPCQA) {
  //     histos.fill(HIST("Lambda/h3dPosNsigmaTPC"), centrality, lambda.pt(), lambdaPosTrackExtra.tpcNSigmaPr());
  //     histos.fill(HIST("Lambda/h3dNegNsigmaTPC"), centrality, lambda.pt(), lambdaNegTrackExtra.tpcNSigmaPi());
  //     histos.fill(HIST("Lambda/h3dPosTPCsignal"), centrality, lambda.pt(), lambdaPosTrackExtra.tpcSignal());
  //     histos.fill(HIST("Lambda/h3dNegTPCsignal"), centrality, lambda.pt(), lambdaNegTrackExtra.tpcSignal());
  //     histos.fill(HIST("Lambda/h3dPosNsigmaTPCvsTrackPtot"), centrality, lambda.positivept() * TMath::CosH(lambda.positiveeta()), lambdaPosTrackExtra.tpcNSigmaPr());
  //     histos.fill(HIST("Lambda/h3dNegNsigmaTPCvsTrackPtot"), centrality, lambda.negativept() * TMath::CosH(lambda.negativeeta()), lambdaNegTrackExtra.tpcNSigmaPi());
  //     histos.fill(HIST("Lambda/h3dPosTPCsignalVsTrackPtot"), centrality, lambda.positivept() * TMath::CosH(lambda.positiveeta()), lambdaPosTrackExtra.tpcSignal());
  //     histos.fill(HIST("Lambda/h3dNegTPCsignalVsTrackPtot"), centrality, lambda.negativept() * TMath::CosH(lambda.negativeeta()), lambdaNegTrackExtra.tpcSignal());
  //     histos.fill(HIST("Lambda/h3dPosNsigmaTPCvsTrackPt"), centrality, lambda.positivept(), lambdaPosTrackExtra.tpcNSigmaPr());
  //     histos.fill(HIST("Lambda/h3dNegNsigmaTPCvsTrackPt"), centrality, lambda.negativept(), lambdaNegTrackExtra.tpcNSigmaPi());
  //     histos.fill(HIST("Lambda/h3dPosTPCsignalVsTrackPt"), centrality, lambda.positivept(), lambdaPosTrackExtra.tpcSignal());
  //     histos.fill(HIST("Lambda/h3dNegTPCsignalVsTrackPt"), centrality, lambda.negativept(), lambdaNegTrackExtra.tpcSignal());
  //   }
  //   if (doTOFQA) {
  //     histos.fill(HIST("Lambda/h3dPosTOFdeltaT"), centrality, lambda.pt(), lambda.posTOFDeltaTLaPr());
  //     histos.fill(HIST("Lambda/h3dNegTOFdeltaT"), centrality, lambda.pt(), lambda.negTOFDeltaTLaPi());
  //     histos.fill(HIST("Lambda/h3dPosTOFdeltaTvsTrackPtot"), centrality, lambda.positivept() * TMath::CosH(lambda.positiveeta()), lambda.posTOFDeltaTLaPr());
  //     histos.fill(HIST("Lambda/h3dNegTOFdeltaTvsTrackPtot"), centrality, lambda.negativept() * TMath::CosH(lambda.negativeeta()), lambda.negTOFDeltaTLaPi());
  //     histos.fill(HIST("Lambda/h3dPosTOFdeltaTvsTrackPt"), centrality, lambda.positivept(), lambda.posTOFDeltaTLaPr());
  //     histos.fill(HIST("Lambda/h3dNegTOFdeltaTvsTrackPt"), centrality, lambda.negativept(), lambda.negTOFDeltaTLaPi());
  //   } 

  //   // Anti Lambda part
  //   auto antiLambdaPosTrackExtra = antiLambda.template posTrackExtra_as<dauTracks>();
  //   auto antiLambdaNegTrackExtra = antiLambda.template negTrackExtra_as<dauTracks>();

  //   bool antiLambdaPosIsFromAfterburner = antiLambdaPosTrackExtra.itsChi2PerNcl() < 0;
  //   bool antiLambdaNegIsFromAfterburner = antiLambdaNegTrackExtra.itsChi2PerNcl() < 0;

  //   uint antiLambdaPosDetMap = computeDetBitmap(antiLambdaPosTrackExtra.detectorMap());
  //   int antiLambdaPosITSclusMap = computeITSclusBitmap(antiLambdaPosTrackExtra.itsClusterMap(), antiLambdaPosIsFromAfterburner);
  //   uint antiLambdaNegDetMap = computeDetBitmap(antiLambdaNegTrackExtra.detectorMap());
  //   int antiLambdaNegITSclusMap = computeITSclusBitmap(antiLambdaNegTrackExtra.itsClusterMap(), antiLambdaNegIsFromAfterburner);
  //   // __________________________________________
  //   // main analysis
  //   if (doPlainTopoQA) {
  //     histos.fill(HIST("AntiLambda/hInvMassWindow"), antiLambda.mAntiLambda()-pdgDB->Mass(3122));
  //     histos.fill(HIST("AntiLambda/hPosDCAToPV"), antiLambda.dcapostopv());
  //     histos.fill(HIST("AntiLambda/hNegDCAToPV"), antiLambda.dcanegtopv());
  //     histos.fill(HIST("AntiLambda/hDCADaughters"), antiLambda.dcaV0daughters());
  //     histos.fill(HIST("AntiLambda/hPointingAngle"), TMath::ACos(antiLambda.v0cosPA()));
  //     histos.fill(HIST("AntiLambda/hV0Radius"), antiLambda.v0radius());
  //     histos.fill(HIST("AntiLambda/h2dArmenterosSelected"), antiLambda.alpha(), antiLambda.qtarm()); // cross-check
  //     histos.fill(HIST("AntiLambda/h2dPositiveITSvsTPCpts"), antiLambdaPosTrackExtra.tpcCrossedRows(), antiLambdaPosTrackExtra.itsNCls());
  //     histos.fill(HIST("AntiLambda/h2dNegativeITSvsTPCpts"), antiLambdaNegTrackExtra.tpcCrossedRows(), antiLambdaNegTrackExtra.itsNCls());
  //   }
  //   if (doDetectPropQA) {
  //     histos.fill(HIST("AntiLambda/h6dDetectPropVsCentrality"), centrality, antiLambdaPosDetMap, antiLambdaPosITSclusMap, antiLambdaNegDetMap, antiLambdaNegITSclusMap, antiLambda.pt());
  //     histos.fill(HIST("AntiLambda/h4dPosDetectPropVsCentrality"), centrality, antiLambdaPosTrackExtra.detectorMap(), antiLambdaPosTrackExtra.itsClusterMap(), antiLambda.pt());
  //     histos.fill(HIST("AntiLambda/h4dNegDetectPropVsCentrality"), centrality, antiLambdaNegTrackExtra.detectorMap(), antiLambdaNegTrackExtra.itsClusterMap(), antiLambda.pt());
  //   }
  //   if (doTPCQA) {
  //     histos.fill(HIST("AntiLambda/h3dPosNsigmaTPC"), centrality, antiLambda.pt(), antiLambdaPosTrackExtra.tpcNSigmaPr());
  //     histos.fill(HIST("AntiLambda/h3dNegNsigmaTPC"), centrality, antiLambda.pt(), antiLambdaNegTrackExtra.tpcNSigmaPi());
  //     histos.fill(HIST("AntiLambda/h3dPosTPCsignal"), centrality, antiLambda.pt(), antiLambdaPosTrackExtra.tpcSignal());
  //     histos.fill(HIST("AntiLambda/h3dNegTPCsignal"), centrality, antiLambda.pt(), antiLambdaNegTrackExtra.tpcSignal());
  //     histos.fill(HIST("AntiLambda/h3dPosNsigmaTPCvsTrackPtot"), centrality, antiLambda.positivept() * TMath::CosH(antiLambda.positiveeta()), antiLambdaPosTrackExtra.tpcNSigmaPr());
  //     histos.fill(HIST("AntiLambda/h3dNegNsigmaTPCvsTrackPtot"), centrality, antiLambda.negativept() * TMath::CosH(antiLambda.negativeeta()), antiLambdaNegTrackExtra.tpcNSigmaPi());
  //     histos.fill(HIST("AntiLambda/h3dPosTPCsignalVsTrackPtot"), centrality, antiLambda.positivept() * TMath::CosH(antiLambda.positiveeta()), antiLambdaPosTrackExtra.tpcSignal());
  //     histos.fill(HIST("AntiLambda/h3dNegTPCsignalVsTrackPtot"), centrality, antiLambda.negativept() * TMath::CosH(antiLambda.negativeeta()), antiLambdaNegTrackExtra.tpcSignal());
  //     histos.fill(HIST("AntiLambda/h3dPosNsigmaTPCvsTrackPt"), centrality, antiLambda.positivept(), antiLambdaPosTrackExtra.tpcNSigmaPr());
  //     histos.fill(HIST("AntiLambda/h3dNegNsigmaTPCvsTrackPt"), centrality, antiLambda.negativept(), antiLambdaNegTrackExtra.tpcNSigmaPi());
  //     histos.fill(HIST("AntiLambda/h3dPosTPCsignalVsTrackPt"), centrality, antiLambda.positivept(), antiLambdaPosTrackExtra.tpcSignal());
  //     histos.fill(HIST("AntiLambda/h3dNegTPCsignalVsTrackPt"), centrality, antiLambda.negativept(), antiLambdaNegTrackExtra.tpcSignal());
  //   }
  //   if (doTOFQA) {
  //     histos.fill(HIST("AntiLambda/h3dPosTOFdeltaT"), centrality, antiLambda.pt(), antiLambda.posTOFDeltaTLaPr());
  //     histos.fill(HIST("AntiLambda/h3dNegTOFdeltaT"), centrality, antiLambda.pt(), antiLambda.negTOFDeltaTLaPi());
  //     histos.fill(HIST("AntiLambda/h3dPosTOFdeltaTvsTrackPtot"), centrality, antiLambda.positivept() * TMath::CosH(antiLambda.positiveeta()), antiLambda.posTOFDeltaTLaPr());
  //     histos.fill(HIST("AntiLambda/h3dNegTOFdeltaTvsTrackPtot"), centrality, antiLambda.negativept() * TMath::CosH(antiLambda.negativeeta()), antiLambda.negTOFDeltaTLaPi());
  //     histos.fill(HIST("AntiLambda/h3dPosTOFdeltaTvsTrackPt"), centrality, antiLambda.positivept(), antiLambda.posTOFDeltaTLaPr());
  //     histos.fill(HIST("AntiLambda/h3dNegTOFdeltaTvsTrackPt"), centrality, antiLambda.negativept(), antiLambda.negTOFDeltaTLaPi());
  //   } 
  // }

  template <typename TV0>
  void analyseCandidate(TV0 lambda, TV0 antiLambda, float centrality, uint8_t gapSide)
  // fill information related to the quarkonium mother
  {
    float pt = RecoDecay::pt(lambda.px() + antiLambda.px(), lambda.py() + antiLambda.py());
    float invmass = RecoDecay::m(std::array{std::array{lambda.px(), lambda.py(), lambda.pz()}, std::array{antiLambda.px(), antiLambda.py(), antiLambda.pz()}}, std::array{o2::constants::physics::MassLambda0, o2::constants::physics::MassLambda0Bar});
    float rapidity = RecoDecay::y(std::array{lambda.px() + antiLambda.px(), lambda.py() + antiLambda.py(), lambda.pz() + antiLambda.pz()}, invmass);
    // rapidity cut on the quarkonium mother
    if (TMath::Abs(rapidity) > rapidityQuarkoniumCut)
      return;

    // fillV0sInfo(lambda, antiLambda, centrality);
      
    // __________________________________________
    // main analysis  
    histos.fill(HIST("h3dMassDiV0s"), centrality, pt, invmass);
    if (!isPP) { // in case of PbPb data
      if (gapSide == 0)
        histos.fill(HIST("h3dMassDiV0sSGA"), centrality, pt, invmass);
      else if (gapSide == 1)
        histos.fill(HIST("h3dMassDiV0sSGC"), centrality, pt, invmass);
      else if (gapSide == 2)
        histos.fill(HIST("h3dMassDiV0sDG"), centrality, pt, invmass);
      else
        histos.fill(HIST("h3dMassDiV0sHadronic"), centrality, pt, invmass);
    }
  }

  // ______________________________________________________
  // Real data processing - no MC subscription
  void processRealData(soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraStamps>::iterator const& collision, v0Candidates const& fullV0s, dauTracks const&)
  {
    // Fire up CCDB
    if ((mlConfigurations.useK0ShortScores && mlConfigurations.calculateK0ShortScores) ||
        (mlConfigurations.useLambdaScores && mlConfigurations.calculateLambdaScores) ||
        (mlConfigurations.useAntiLambdaScores && mlConfigurations.calculateAntiLambdaScores)) {
      initCCDB(collision);
    }

    histos.fill(HIST("hEventSelection"), 0. /* all collisions */);
    if (requireSel8 && !collision.sel8()) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 1 /* sel8 collisions */);

    if (std::abs(collision.posZ()) > 10.f) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 2 /* vertex-Z selected */);

    if (rejectITSROFBorder && !collision.selection_bit(o2::aod::evsel::kNoITSROFrameBorder)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 3 /* Not at ITS ROF border */);

    if (rejectTFBorder && !collision.selection_bit(o2::aod::evsel::kNoTimeFrameBorder)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 4 /* Not at TF border */);

    if (requireIsVertexITSTPC && !collision.selection_bit(o2::aod::evsel::kIsVertexITSTPC)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 5 /* Contains at least one ITS-TPC track */);

    if (requireIsGoodZvtxFT0VsPV && !collision.selection_bit(o2::aod::evsel::kIsGoodZvtxFT0vsPV)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 6 /* PV position consistency check */);

    if (requireIsVertexTOFmatched && !collision.selection_bit(o2::aod::evsel::kIsVertexTOFmatched)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 7 /* PV with at least one contributor matched with TOF */);

    if (requireIsVertexTRDmatched && !collision.selection_bit(o2::aod::evsel::kIsVertexTRDmatched)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 8 /* PV with at least one contributor matched with TRD */);

    if (rejectSameBunchPileup && !collision.selection_bit(o2::aod::evsel::kNoSameBunchPileup)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 9 /* Not at same bunch pile-up */);

    if (requireNoCollInTimeRangeStd && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeStandard)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 10 /* No other collision within +/- 10 microseconds */);

    if (requireNoCollInTimeRangeNarrow && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeNarrow)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 11 /* No other collision within +/- 4 microseconds */);

    if (minOccupancy > 0 && collision.trackOccupancyInTimeRange() < minOccupancy) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 12 /* Below min occupancy */);
    if (maxOccupancy > 0 && collision.trackOccupancyInTimeRange() > maxOccupancy) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 13 /* Above max occupancy */);

    float centrality = -1;
    if (isPP) { //
      centrality = collision.centFT0M(); 

      if (qaCentrality) {
        auto hRawCentrality = histos.get<TH1>(HIST("hRawCentrality"));
        centrality = hRawCentrality->GetBinContent(hRawCentrality->FindBin(collision.multFT0A() + collision.multFT0C()));
      }
    } else {
      centrality = collision.centFT0C();

      if (qaCentrality) {
        auto hRawCentrality = histos.get<TH1>(HIST("hRawCentrality"));
        centrality = hRawCentrality->GetBinContent(hRawCentrality->FindBin(collision.multFT0C()));
      }
    }

    // gap side
    int gapSide = collision.gapSide();
    int selGapSide = -1;
    if (!isPP) {
      // -1 --> Hadronic
      // 0 --> Single Gap - A side
      // 1 --> Single Gap - C side
      // 2 --> Double Gap - both A & C sides
      selGapSide = sgSelector.trueGap(collision, upcCuts.FV0cut, upcCuts.FT0Acut, upcCuts.FT0Ccut, upcCuts.ZDCcut);
      histos.fill(HIST("hGapSide"), gapSide);
      histos.fill(HIST("hSelGapSide"), selGapSide);
      histos.fill(HIST("hEventCentralityVsSelGapSide"), centrality, selGapSide <= 2 ? selGapSide : -1);
    }

    histos.fill(HIST("hEventCentrality"), centrality);

    histos.fill(HIST("hCentralityVsNch"), centrality, collision.multNTracksPVeta1());

    histos.fill(HIST("hEventOccupancy"), collision.trackOccupancyInTimeRange());
    histos.fill(HIST("hCentralityVsOccupancy"), centrality, collision.trackOccupancyInTimeRange());

    // __________________________________________
    // perform main analysis
    std::vector<bool> selK0ShortIndices(fullV0s.size());
    std::vector<bool> selLambdaIndices(fullV0s.size());
    std::vector<bool> selAntiLambdaIndices(fullV0s.size());
    for (auto& v0 : fullV0s) {
      if (std::abs(v0.negativeeta()) > v0Selections.daughterEtaCut || std::abs(v0.positiveeta()) > v0Selections.daughterEtaCut)
        continue; // remove acceptance that's badly reproduced by MC / superfluous in future

      if (v0.v0Type() != v0Selections.v0TypeSelection && v0Selections.v0TypeSelection > -1)
        continue; // skip V0s that are not standard

      uint64_t selMap = computeReconstructionBitmap(v0, collision, v0.yLambda(), v0.yK0Short(), v0.pt());

      // consider for histograms for all species
      selMap = selMap | (uint64_t(1) << selConsiderK0Short) | (uint64_t(1) << selConsiderLambda) | (uint64_t(1) << selConsiderAntiLambda);
      selMap = selMap | (uint64_t(1) << selPhysPrimK0Short) | (uint64_t(1) << selPhysPrimLambda) | (uint64_t(1) << selPhysPrimAntiLambda);

      analyseV0Candidate(v0, v0.pt(), centrality, selMap, selK0ShortIndices, selLambdaIndices, selAntiLambdaIndices, fullV0s.offset());
    } // end v0 loop

    // count the number of K0s, Lambda and AntiLambdas passsing the selections
    int nK0Shorts = std::count(selK0ShortIndices.begin(), selK0ShortIndices.end(), true);
    int nLambdas = std::count(selLambdaIndices.begin(), selLambdaIndices.end(), true);
    int nAntiLambdas = std::count(selAntiLambdaIndices.begin(), selAntiLambdaIndices.end(), true);

    // fill the histograms with the number of reconstructed K0s/Lambda/antiLambda per collision
    histos.fill(HIST("h2dNbrOfK0ShortVsCentrality"), centrality, nK0Shorts);
    histos.fill(HIST("h2dNbrOfLambdaVsCentrality"), centrality, nLambdas);
    histos.fill(HIST("h2dNbrOfAntiLambdaVsCentrality"), centrality, nAntiLambdas);

    // Check the number of Lambdas and antiLambdas
    // if not at least 1 of each, we stop here
    if (nLambdas < 1 || nAntiLambdas < 1) {
      return;
    } 

    // 1st loop over all v0s
    for (auto& lambda : fullV0s) { 
      // select only v0s matching Lambda selections
      if (!selLambdaIndices[lambda.globalIndex() - fullV0s.offset()]) { // local index needed due to collisions grouping
        continue;
      }

      // 2nd loop over all v0s
      for (auto& antiLambda : fullV0s) {
        // select only v0s matching Anti-Lambda selections
        if (!selLambdaIndices[antiLambda.globalIndex() - fullV0s.offset()]) { // local index needed due to collisions grouping
          continue;
        }

        // check we don't look at the same v0s
        if (lambda.globalIndex() == antiLambda.globalIndex()) {
          continue;
        }

        // form V0 pairs and fill histograms
        analyseCandidate(lambda, antiLambda, centrality, selGapSide);
      } // end antiLambda loop
    } // end lambda loop

    return;
  }

  // ______________________________________________________
  // Simulated processing (subscribes to MC information too)
  void processMonteCarlo(soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraCollLabels>::iterator const& collision, v0MCCandidates const& fullV0s, dauTracks const&, aod::MotherMCParts const&, soa::Join<aod::StraMCCollisions, aod::StraMCCollMults> const& /*mccollisions*/, soa::Join<aod::V0MCCores, aod::V0MCCollRefs> const&)
  {
    histos.fill(HIST("hEventSelection"), 0. /* all collisions */);
    if (requireSel8 && !collision.sel8()) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 1 /* sel8 collisions */);

    if (std::abs(collision.posZ()) > 10.f) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 2 /* vertex-Z selected */);

    if (rejectITSROFBorder && !collision.selection_bit(o2::aod::evsel::kNoITSROFrameBorder)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 3 /* Not at ITS ROF border */);

    if (rejectTFBorder && !collision.selection_bit(o2::aod::evsel::kNoTimeFrameBorder)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 4 /* Not at TF border */);

    if (requireIsVertexITSTPC && !collision.selection_bit(o2::aod::evsel::kIsVertexITSTPC)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 5 /* Contains at least one ITS-TPC track */);

    if (requireIsGoodZvtxFT0VsPV && !collision.selection_bit(o2::aod::evsel::kIsGoodZvtxFT0vsPV)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 6 /* PV position consistency check */);

    if (requireIsVertexTOFmatched && !collision.selection_bit(o2::aod::evsel::kIsVertexTOFmatched)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 7 /* PV with at least one contributor matched with TOF */);

    if (requireIsVertexTRDmatched && !collision.selection_bit(o2::aod::evsel::kIsVertexTRDmatched)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 8 /* PV with at least one contributor matched with TRD */);

    if (rejectSameBunchPileup && !collision.selection_bit(o2::aod::evsel::kNoSameBunchPileup)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 9 /* Not at same bunch pile-up */);

    if (requireNoCollInTimeRangeStd && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeStandard)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 10 /* No other collision within +/- 10 microseconds */);

    if (requireNoCollInTimeRangeNarrow && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeNarrow)) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 11 /* No other collision within +/- 4 microseconds */);

    if (minOccupancy > 0 && collision.trackOccupancyInTimeRange() < minOccupancy) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 12 /* Below min occupancy */);
    if (maxOccupancy > 0 && collision.trackOccupancyInTimeRange() > maxOccupancy) {
      return;
    }
    histos.fill(HIST("hEventSelection"), 13 /* Above max occupancy */);

    float centrality = -1;
    if (isPP) { //
      centrality = collision.centFT0M(); 

      if (qaCentrality) {
        auto hRawCentrality = histos.get<TH1>(HIST("hRawCentrality"));
        centrality = hRawCentrality->GetBinContent(hRawCentrality->FindBin(collision.multFT0A() + collision.multFT0C()));
      }
    } else {
      centrality = collision.centFT0C();

      if (qaCentrality) {
        auto hRawCentrality = histos.get<TH1>(HIST("hRawCentrality"));
        centrality = hRawCentrality->GetBinContent(hRawCentrality->FindBin(collision.multFT0C()));
      }

    }

    // gap side
    int gapSide = collision.gapSide();
    int selGapSide = -1;
    if (!isPP) {
      // -1 --> Hadronic
      // 0 --> Single Gap - A side
      // 1 --> Single Gap - C side
      // 2 --> Double Gap - both A & C sides
      selGapSide = sgSelector.trueGap(collision, upcCuts.FV0cut, upcCuts.FT0Acut, upcCuts.FT0Ccut, upcCuts.ZDCcut);
      histos.fill(HIST("hGapSide"), gapSide);
      histos.fill(HIST("hSelGapSide"), selGapSide);
      histos.fill(HIST("hEventCentralityVsSelGapSide"), centrality, selGapSide <= 2 ? selGapSide : -1);
    }

    histos.fill(HIST("hEventCentrality"), centrality);

    histos.fill(HIST("hCentralityVsNch"), centrality, collision.multNTracksPVeta1());

    histos.fill(HIST("hEventOccupancy"), collision.trackOccupancyInTimeRange());
    histos.fill(HIST("hCentralityVsOccupancy"), centrality, collision.trackOccupancyInTimeRange());

    // __________________________________________
    // perform main analysis
    std::vector<bool> selK0ShortIndices(fullV0s.size());
    std::vector<bool> selLambdaIndices(fullV0s.size());
    std::vector<bool> selAntiLambdaIndices(fullV0s.size());
    for (auto& v0 : fullV0s) {
      if (std::abs(v0.negativeeta()) > v0Selections.daughterEtaCut || std::abs(v0.positiveeta()) > v0Selections.daughterEtaCut)
        continue; // remove acceptance that's badly reproduced by MC / superfluous in future

      if (!v0.has_v0MCCore())
        continue;

      auto v0MC = v0.v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();

      float ptmc = RecoDecay::sqrtSumOfSquares(v0MC.pxPosMC() + v0MC.pxNegMC(), v0MC.pyPosMC() + v0MC.pyNegMC());
      float ymc = 1e-3;
      if (v0MC.pdgCode() == 310)
        ymc = RecoDecay::y(std::array{v0MC.pxPosMC() + v0MC.pxNegMC(), v0MC.pyPosMC() + v0MC.pyNegMC(), v0MC.pzPosMC() + v0MC.pzNegMC()}, o2::constants::physics::MassKaonNeutral);
      else if (TMath::Abs(v0MC.pdgCode()) == 3122)
        ymc = RecoDecay::y(std::array{v0MC.pxPosMC() + v0MC.pxNegMC(), v0MC.pyPosMC() + v0MC.pyNegMC(), v0MC.pzPosMC() + v0MC.pzNegMC()}, o2::constants::physics::MassLambda);

      uint64_t selMap = computeReconstructionBitmap(v0, collision, ymc, ymc, ptmc);
      selMap = selMap | computeMCAssociation(v0MC);

      // feeddown matrix always with association
      // if (calculateFeeddownMatrix)
        // fillFeeddownMatrix(v0, ptmc, centrality, selMap);

      // consider only associated candidates if asked to do so, disregard association
      if (!doMCAssociation) {
        selMap = selMap | (uint64_t(1) << selConsiderK0Short) | (uint64_t(1) << selConsiderLambda) | (uint64_t(1) << selConsiderAntiLambda);
        selMap = selMap | (uint64_t(1) << selPhysPrimK0Short) | (uint64_t(1) << selPhysPrimLambda) | (uint64_t(1) << selPhysPrimAntiLambda);
      }

      analyseV0Candidate(v0, ptmc, centrality, selMap, selK0ShortIndices, selLambdaIndices, selAntiLambdaIndices, fullV0s.offset());
    } // end v0 loop

    /// count the number of K0s, Lambda and AntiLambdas passsing the selections
    int nK0Shorts = std::count(selK0ShortIndices.begin(), selK0ShortIndices.end(), true);
    int nLambdas = std::count(selLambdaIndices.begin(), selLambdaIndices.end(), true);
    int nAntiLambdas = std::count(selAntiLambdaIndices.begin(), selAntiLambdaIndices.end(), true);

    // fill the histograms with the number of reconstructed K0s/Lambda/antiLambda per collision
    histos.fill(HIST("h2dNbrOfK0ShortVsCentrality"), centrality, nK0Shorts);
    histos.fill(HIST("h2dNbrOfLambdaVsCentrality"), centrality, nLambdas);
    histos.fill(HIST("h2dNbrOfAntiLambdaVsCentrality"), centrality, nAntiLambdas);

    // Check the number of Lambdas and antiLambdas
    // if not at least 1 of each, we stop here
    if (nLambdas < 1 || nAntiLambdas < 1) {
      return;
    } 

    // 1st loop over all v0s
    for (auto& lambda : fullV0s) { 
      // select only v0s matching Lambda selections
      if (!selLambdaIndices[lambda.globalIndex() - fullV0s.offset()]) { // local index needed due to collisions grouping
        continue;
      }

      // 2nd loop over all v0s
      for (auto& antiLambda : fullV0s) {
        // select only v0s matching Anti-Lambda selections
        if (!selLambdaIndices[antiLambda.globalIndex() - fullV0s.offset()]) { // local index needed due to collisions grouping
          continue;
        }

        // check we don't look at the same v0s
        if (lambda.globalIndex() == antiLambda.globalIndex()) {
          continue;
        }

        // form V0 pairs and fill histograms
        analyseCandidate(lambda, antiLambda, centrality, selGapSide);

        // if (doCollisionAssociationQA) {
        //   // check collision association explicitly
        //   auto lambdaMC = lambda.v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();
        //   auto antiLambdaMC = antiLambda.v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();

        //   bool correctCollisionLambda = false;
        //   bool correctCollisionAntiLambda = false;
        //   int mcNch = -1;

        //   if (collision.has_straMCCollision()) {
        //     auto mcCollision = collision.straMCCollision_as<soa::Join<aod::StraMCCollisions, aod::StraMCCollMults>>();
        //     mcNch = mcCollision.multMCNParticlesEta05();

        //     correctCollisionLambda = (lambdaMC.straMCCollisionId() == mcCollision.globalIndex());
        //     correctCollisionAntiLambda = (antiLambdaMC.straMCCollisionId() == mcCollision.globalIndex());
        //   }
        //   analyseCollisionAssociation(lambdaMC, antiLambdaMC, mcNch, correctCollisionLambda, correctCollisionAntiLambda);
        // }
      } // end antiLambda loop
    } // end lambda loop
  }

  PROCESS_SWITCH(derivedquarkoniaanalysis, processRealData, "process as if real data", true);
  PROCESS_SWITCH(derivedquarkoniaanalysis, processMonteCarlo, "process as if MC", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<derivedquarkoniaanalysis>(cfgc)};
}
