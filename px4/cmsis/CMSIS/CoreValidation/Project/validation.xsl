<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" indent="yes"/>
    <xsl:template match="/">
        <testsuites>
            <xsl:variable name="buildName" select="//report/test/title"/>
            <xsl:variable name="numberOfTests" select="//report/test/summary/tcnt"/>
            <xsl:variable name="numberOfExecutes" select="//report/test/summary/exec"/>
            <xsl:variable name="numberOfPasses" select="//report/test/summary/pass"/>
            <xsl:variable name="numberOfFailures" select="//report/test/summary/fail"/>
			<xsl:variable name="numberOfSkips" select="$numberOfTests - $numberOfExecutes"/>
			<xsl:variable name="numberOfErrors" select="0"/>
            <testsuite name="{$buildName}"
                       tests="{$numberOfTests}" time="0"
                       failures="{$numberOfFailures}" errors="{$numberOfErrors}"
                       skipped="{$numberOfSkips}">
                <xsl:for-each select="//report/test/test_cases/tc">
                    <xsl:variable name="testName" select="func"/>
                    <xsl:variable name="status" select="res"/>
                    <testcase name="{$testName}">
                        <xsl:choose>
                            <xsl:when test="res='PASSED'"/>
							<xsl:when test="res='NOT EXECUTED'">
								<skipped/>
							</xsl:when>
                            <xsl:otherwise>
                                <failure>
                                    <xsl:for-each select="dbgi/detail">
                                        <xsl:variable name="file" select="module"/>
                                        <xsl:variable name="line" select="line"/>
                                        <xsl:text>&#10;        </xsl:text>
                                        <xsl:value-of select="$file"/>:<xsl:value-of select="$line"/>
                                    </xsl:for-each>
                                    <xsl:text>&#10;      </xsl:text>
                                 </failure>
                            </xsl:otherwise>
                        </xsl:choose>
                    </testcase>
                </xsl:for-each>
            </testsuite>
        </testsuites>
    </xsl:template>
</xsl:stylesheet>
